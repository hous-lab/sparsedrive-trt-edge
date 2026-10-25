#!/usr/bin/env python
"""x86 PT-vs-TRT gate for motion/planning (v2 engine, engine-only).

Compares the x86 FP16 engine cascade outputs (data/x86_reference.npz, keys
mp_*) against the PT motion-head outputs captured at demo-data generation
(data/demo_data/sample0.npz, keys pt_*) on the same frame with the same
deployed bridge (map_reg zeroed, det bridge from the PT det head).

Usage: python x86/eval/eval_motion_v2_x86.py
Env: SD_DEMO_NPZ, SD_REF_NPZ
"""
import os, json
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEMO = os.environ.get('SD_DEMO_NPZ', os.path.join(REPO, 'data/demo_data/sample0.npz'))
REF = os.environ.get('SD_REF_NPZ', os.path.join(REPO, 'data/x86_reference.npz'))

GATES = {
    'mp_motion_cls': ('pt_motion_cls', 0.95),
    'mp_motion_traj': ('pt_motion_traj', 0.95),
    'mp_traj_cls': ('pt_traj_cls', 0.90),
    'mp_lat_anchor': ('pt_lat_anchor', 0.99),
    'mp_lon_anchor': ('pt_lon_anchor', 0.99),
}


def cos_sim(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def mse_v(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(((a - b) ** 2).mean())


def _sanitize(obj):
    """Record paths repo-relative so committed results carry no host paths."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, str) and obj.startswith(REPO + '/'):
        return os.path.relpath(obj, REPO)
    return obj


def main():
    demo = np.load(DEMO)
    ref = np.load(REF)
    print('PT (demo npz) vs x86 FP16 engines (ref npz), same frame:')
    print(f"{'output':<18s} {'cos_sim':<12s} {'mse':<12s} gate")
    print('-' * 52)
    results = {}
    ok = True
    for trt_key, (pt_key, gate) in GATES.items():
        a = ref[trt_key]
        b = demo[pt_key]
        if a.shape != b.shape:
            print(f'{trt_key:<18s} SHAPE {a.shape} vs PT {b.shape}')
            ok = False
            continue
        cs = cos_sim(a, b)
        ms = mse_v(a, b)
        st = 'OK' if cs >= gate else 'WARN' if cs >= gate - 0.02 else 'FAIL'
        if st == 'FAIL':
            ok = False
        print(f'{trt_key:<18s} {cs:<12.6f} {ms:<12.8f} {st} (gate {gate})')
        results[trt_key] = {'cos_sim': cs, 'mse': ms, 'status': st, 'gate': gate}
    out = os.path.join(REPO, 'x86/eval/results/motion_v2_gate.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(_sanitize({'demo': DEMO, 'reference': REF, 'results': results,
                         'gates_pass': ok}), f, indent=2)
    print('Saved: ' + out)
    print('GATES PASS: %s' % ok)


if __name__ == '__main__':
    main()
