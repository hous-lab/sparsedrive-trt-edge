"""Orin accuracy check: run Orin engines on the x86 reference inputs.

Feeds the exact x86 FP16 cascade inputs (data/x86_reference.npz) through the
Orin FP16 / BB-INT8 engine sets and compares every output against the
reference. NaN in any output fails the config outright.

Usage (on Orin): python arm64/eval/eval_accuracy_orin.py
Env: SD_REF_NPZ, SD_ENGINE_DIR, SD_ENGINE_INT8_DIR, SD_PLUGIN_LIB
"""
import os, ctypes, json
import numpy as np
import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
PLUGIN = os.environ.get(
    "SD_PLUGIN_LIB", os.path.join(REPO, "arm64/plugins_build/build/libplugins_orin.so"))
REF = os.environ.get("SD_REF_NPZ", os.path.join(REPO, "data/x86_reference.npz"))
ENGINE_FP16_DIR = os.environ.get(
    "SD_ENGINE_DIR", os.path.join(REPO, "arm64/trt_compile/fp16/output"))
ENGINE_INT8_DIR = os.environ.get(
    "SD_ENGINE_INT8_DIR", os.path.join(REPO, "arm64/trt_compile/int8/output"))

FEAT_SHAPES = [(96, 176), (48, 88), (24, 44), (12, 22)]
# Mirrors GATE_OVERRIDES in tools/compare_npz_gates.py - one gate policy
# across platforms (x86 measured mp_lon_anchor 0.9742 even on pure FP16,
# docs/lessons.md 7.6; INT8 backbone lands ~0.98 on bb_feat by design).
GATES = {
    "bb_feat_0": 0.98, "bb_feat_1": 0.98,
    "bb_feat_2": 0.98, "bb_feat_3": 0.98,
    "hd_det_cls": 0.99, "hd_det_reg": 0.99,
    "hd_det_quality": 0.98, "hd_instance_feature": 0.99,
    "map_cls": 0.99, "map_instance_feat": 0.99, "map_anchor_embed": 0.99,
    "mp_motion_cls": 0.95, "mp_motion_traj": 0.95,
    "mp_traj_cls": 0.95, "mp_lat_anchor": 0.99, "mp_lon_anchor": 0.95,
}



def _flatten(feats):
    # deformable layout: tokens = per level 6 cams x h*w, channel-last, cat over
    # levels -> [1, N_tokens, 256] (see docs/lessons.md §7.4 / §2 of pipeline guide)
    flat = torch.cat([f.reshape(f.shape[0], f.shape[1], -1).permute(0, 2, 1).reshape(-1, f.shape[1])
                      for f in feats], dim=0)
    # Engine binding is 4-D (1, n_cam, sum_hw, 256); the cat yields the same
    # bytes as (6*sum_hw, 256) level-major/channel-last, so this reshape is
    # metadata-only - it satisfies TRT's static shape check without touching
    # memory. Feeding the 3-D tensor still works (same element count) but TRT
    # prints a setInputShape error per call.
    n_cam = feats[0].shape[0]
    flat = flat.view(1, n_cam, -1, feats[0].shape[1])
    sizes = [h * w for h, w in FEAT_SHAPES]
    spat = torch.tensor([[h, w] for h, w in FEAT_SHAPES], dtype=torch.int32)
    # ONNX lvl_start is [num_levels]=4 (start offsets); cumsum gives a trailing
    # sentinel the engine does not want - slice it off or TRT rejects the shape.
    lvl = torch.tensor(np.cumsum([0] + sizes)[:-1], dtype=torch.int32)
    return flat.contiguous().cuda(), spat.cuda(), lvl.cuda()


def np_to_cuda(key, arr):
    a = np.ascontiguousarray(arr)
    if a.dtype == np.float64 or a.dtype == np.float16:
        a = a.astype(np.float32)
    return torch.from_numpy(a).cuda()


class OrinRunner:
    """Minimal TRT runner: load engine, bind feeds, execute, return outputs."""

    def __init__(self, path):
        with open(path, "rb") as f:
            self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.ins, self.outs = [], []
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.ins.append(n)
            else:
                self.outs.append(n)

    def run(self, feeds):
        for n in self.ins:
            t = feeds[n].contiguous()
            self.ctx.set_input_shape(n, tuple(t.shape))
            self.ctx.set_tensor_address(n, t.data_ptr())
        outs = {}
        for n in self.outs:
            shape = tuple(self.ctx.get_tensor_shape(n))
            t = torch.empty(shape, device="cuda", dtype=torch.float32)
            self.ctx.set_tensor_address(n, t.data_ptr())
            outs[n] = t
        self.ctx.execute_async_v3(0)
        torch.cuda.current_stream().synchronize()
        return outs


def run_pipeline(bb_path, head_path, map_path, motion_path, ref):
    """Full cascade on Orin engines; returns prefixed outputs like the ref npz."""
    ctypes.CDLL(PLUGIN, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    bb = OrinRunner(bb_path)
    hd = OrinRunner(head_path)
    mo = OrinRunner(motion_path)
    img = np_to_cuda("img", ref["img"])
    feats = list(bb.run({"img": img}).values())
    feats.sort(key=lambda t: t.shape[-1] * t.shape[-2], reverse=True)
    flat, spat, lvl = _flatten(feats)
    pm = np_to_cuda("projection_mat", ref["projection_mat"])
    iw = np_to_cuda("image_wh", ref["image_wh"])
    hd_out = hd.run({
        "feat_0": feats[0], "feat_1": feats[1],
        "feat_2": feats[2], "feat_3": feats[3],
        "projection_mat": pm, "image_wh": iw})
    mmap = OrinRunner(map_path)
    map_out = mmap.run({
        "feat_0": feats[0], "feat_1": feats[1],
        "feat_2": feats[2], "feat_3": feats[3],
        "projection_mat": pm, "image_wh": iw})
    map_out["map_reg"] = torch.zeros(1, 100, 2, device="cuda")
    mo_feeds = {
        "feat_flat": flat, "spat_shape": spat, "lvl_start": lvl,
        "det_cls": hd_out["det_cls"], "det_reg": hd_out["det_reg"],
        "det_quality": hd_out["det_quality"],
        "det_instance_feat": hd_out["det_instance_feat"],
        "det_anchor_embed": np_to_cuda("a", ref["det_anchor_embed"]).reshape(1, 900, 256),
        "map_cls": map_out["map_cls"],
        "map_reg": map_out["map_reg"],
        "map_instance_feat": map_out["map_instance_feature"],
        "map_anchor_embed": map_out["map_anchor_embed"],
        "projection_mat": pm, "image_wh": iw,
        "timestamp": np_to_cuda("t", ref["timestamp"]),
        "ego_status": np_to_cuda("e", ref["ego_status"]),
        "tp_near": np_to_cuda("n", ref["tp_near"]),
        "gt_ego_fut_cmd": np_to_cuda("g", ref["gt_ego_fut_cmd"]),
        "mask": np_to_cuda("m", ref["mask"]),
        "queue_temp_feat": np_to_cuda("q", ref["queue_temp_feat"]),
        "queue_temp_anchor": np_to_cuda("qa", ref["queue_temp_anchor"]),
        "period": np_to_cuda("p", ref["period"]),
    }
    mo_out = mo.run(mo_feeds)
    out = {"bb_feat_%d" % i: feats[i] for i in range(4)}
    out["hd_det_cls"] = hd_out["det_cls"]
    out["hd_det_reg"] = hd_out["det_reg"]
    out["hd_det_quality"] = hd_out["det_quality"]
    out["hd_instance_feature"] = hd_out["det_instance_feat"]
    out["map_cls"] = map_out["map_cls"]
    out["map_instance_feat"] = map_out["map_instance_feature"]
    out["map_anchor_embed"] = map_out["map_anchor_embed"]
    for k, v in mo_out.items():
        out["mp_" + k] = v
    return out


def cos_sim(a, b):
    a_f = a.flatten().float().cpu().numpy()
    b_f = b.flatten().float().cpu().numpy()
    return float(np.dot(a_f, b_f) /
                 (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12))


def mse(a, b):
    return float(((a.float() - b.float()) ** 2).mean().cpu().numpy())


def compare_config(config_name, bb_path, head_path, map_path, motion_path, ref):
    trt_out = run_pipeline(bb_path, head_path, map_path, motion_path, ref)
    torch.cuda.empty_cache()
    print('\n' + '=' * 60)
    print('  ' + config_name)
    print('=' * 60)
    print(f"{'Output':<24s} {'CosSim':<12s} {'MSE':<12s} Gate")
    print('-' * 56)
    results = {}
    all_pass = True
    for name, gate in GATES.items():
        if name not in trt_out:
            print(f'{name:<24s} MISSING')
            all_pass = False
            continue
        ref_t = torch.from_numpy(np.ascontiguousarray(ref[name])).cuda()
        trt_t = trt_out[name]
        if torch.isnan(trt_t).any():
            print(f'{name:<24s} {"nan":<12s} {"nan":<12s} FAIL (NaN)')
            results[name] = {"cos_sim": float("nan"), "mse": float("nan"),
                             "status": "FAIL", "gate": gate}
            all_pass = False
            continue
        cs = cos_sim(ref_t, trt_t)
        ms = mse(ref_t, trt_t)
        status = "OK" if cs >= gate else "WARN" if cs >= gate - 0.02 else "FAIL"
        if status == "FAIL":
            all_pass = "FAIL"
        print(f'{name:<24s} {cs:<12.6f} {ms:<12.8f} {status} (gate {gate})')
        results[name] = {"cos_sim": cs, "mse": ms, "status": status, "gate": gate}
    return results, all_pass


if __name__ == "__main__":
    ref = np.load(REF)
    configs = [
        ("Config 1: All FP16", "fp16", os.path.join(ENGINE_FP16_DIR, "backbone_fp16.engine"),
         os.path.join(ENGINE_FP16_DIR, "head_fp16.engine"),
         os.path.join(ENGINE_FP16_DIR, "map_fp16.engine"),
         os.path.join(ENGINE_FP16_DIR, "motion_plan_fp16.engine")),
        ("Config 2: BB INT8 + rest FP16", "bb_int8",
         os.path.join(ENGINE_INT8_DIR, "backbone_int8.engine"),
         os.path.join(ENGINE_FP16_DIR, "head_fp16.engine"),
         os.path.join(ENGINE_FP16_DIR, "map_fp16.engine"),
         os.path.join(ENGINE_FP16_DIR, "motion_plan_fp16.engine")),
        ("Config 3: BB+Head INT8 + rest FP16", "bb_head_int8",
         os.path.join(ENGINE_INT8_DIR, "backbone_int8.engine"),
         os.path.join(ENGINE_INT8_DIR, "head_int8.engine"),
         os.path.join(ENGINE_FP16_DIR, "map_fp16.engine"),
         os.path.join(ENGINE_FP16_DIR, "motion_plan_fp16.engine")),
    ]
    all_results = {}
    overall = True
    for cname, ckey, bb, hd, mp, mo in configs:
        if not all(os.path.exists(x) for x in (bb, hd, mp, mo)):
            print('[skip] %s: missing engines' % cname)
            continue
        res, ok = compare_config(cname, bb, hd, mp, mo, ref)
        all_results[ckey] = {
            "config": cname,
            "engines": {"backbone": bb, "head": hd, "map": mp, "motion_plan": mo},
            "results": res,
            "gates_pass": ok is True,
        }
        overall = overall and (ok is True)
    out_path = os.path.join(REPO, "arm64/eval/results/accuracy_comparison.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"reference": REF, "configs": all_results,
                   "gates_pass": overall}, f, indent=2)
    print('\nSaved: %s' % out_path)
    print('GATES PASS: %s' % overall)
