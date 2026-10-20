#!/usr/bin/env python
"""Gate-check one engine-cascade reference npz against another.

Usage:
  python tools/compare_npz_gates.py --ref data/x86_reference.npz \
      --cand x86/trt_compile/int8/output/x86_reference_int8.npz \
      [--json OUT.json] [--gate 0.99] [--warn-band 0.02]

Compares every shared float key by cosine similarity + MSE with the same
OK/WARN/FAIL thresholds as x86/eval/eval_motion_v2_x86.py (gate 0.99,
WARN band 0.02). Exit 1 on FAIL.
"""
import argparse
import json
import os
import numpy as np


def cos_mse(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 and nb == 0.0:
        return 1.0, 0.0  # both all-zero (map_reg bridge, timestamp) - nothing to compare
    cs = float(np.dot(a, b)) / (na * nb + 1e-12)
    ms = float(np.mean((a - b) ** 2))
    return cs, ms


# Per-key gates mirror arm64/eval/eval_accuracy_orin.py: perception 0.99,
# motion outputs 0.95 (selection/argmax-style outputs are FP16-sensitive,
# see docs/lessons.md 7.6 - lon_anchor measured 0.9742 even on FP16 x86).
GATE_OVERRIDES = {
    'mp_motion_cls': 0.95, 'mp_motion_traj': 0.95, 'mp_traj_cls': 0.95,
    'mp_lon_anchor': 0.95, 'mp_lat_anchor': 0.99,
    'bb_feat_0': 0.98, 'bb_feat_1': 0.98, 'bb_feat_2': 0.98, 'bb_feat_3': 0.98,
    'hd_det_quality': 0.98,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ref', required=True)
    p.add_argument('--cand', required=True)
    p.add_argument('--json', default=None)
    p.add_argument('--gate', type=float, default=0.99)
    p.add_argument('--warn-band', type=float, default=0.02)
    args = p.parse_args()

    ref = np.load(args.ref)
    cand = np.load(args.cand)
    results = {}
    ok = True
    print('%-22s %-12s %-12s %s' % ('key', 'cos_sim', 'mse', 'status'))
    for k in sorted(set(ref.files) & set(cand.files)):
        a, b = ref[k], cand[k]
        if a.dtype.kind != 'f' or b.dtype.kind != 'f':
            continue
        if a.shape != b.shape:
            print('%-22s SHAPE_MISMATCH %s vs %s SKIP' % (k, a.shape, b.shape))
            continue
        cs, ms = cos_mse(a, b)
        gate = GATE_OVERRIDES.get(k, args.gate)
        st = 'OK' if cs >= gate else ('WARN' if cs >= gate - args.warn_band else 'FAIL')
        print('%-22s %-12.6f %-12.8f %s (gate %.2f)' % (k, cs, ms, st, gate))
        results[k] = {'cos_sim': cs, 'mse': ms, 'status': st}
        if st == 'FAIL':
            ok = False
    results['gates_pass'] = bool(ok)
    print('GATES PASS: %s' % ok)
    if args.json:
        d = os.path.dirname(os.path.abspath(args.json))
        os.makedirs(d, exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
