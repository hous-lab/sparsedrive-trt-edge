#!/usr/bin/env python
"""Render BEV visualizations from the engine-cascade reference (matplotlib).

Produces three PNGs under docs/assets/: detection + map, per-object motion
modes, and the decoded planning output (path + velocity profile).

Usage: python x86/eval/render_outputs.py
Env: SD_REF_NPZ, SD_OUT_DIR
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
REF = os.environ.get('SD_REF_NPZ', os.path.join(REPO, 'data/x86_reference.npz'))
OUT_DIR = os.environ.get('SD_OUT_DIR', os.path.join(REPO, 'docs/assets'))
MAP_ANCHORS = os.path.join(REPO, 'configs/anchors/kmeans_map_100.npy')


def load_ref():
    r = np.load(REF)
    return r


def render_det_map(r, out_dir):
    cls = r['hd_det_cls'][0]        # [900, 9] logits
    reg = r['hd_det_reg'][0]        # [900, 11] (x,y,z,w,l,h,sin,cos,vx,vy,vz)
    mcls = r['map_cls'][0]          # [100, 6] logits
    manchor = np.load(MAP_ANCHORS)  # [100, 20, 2] ego-frame polylines
    manchor = np.asarray(manchor, dtype=np.float32)
    scores = 1 / (1 + np.exp(-cls.max(axis=-1)))
    keep = scores > 0.25
    score = scores[keep]
    box = reg[keep]
    order = np.argsort(-score)
    score, box = score[order], box[order]
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(len(score)):
        x, y = box[i, 0], box[i, 1]
        w, l = box[i, 3], box[i, 4]
        yaw = np.arctan2(box[i, 6], box[i, 7])
        ca, sa = np.cos(yaw), np.sin(yaw)
        corners = np.array([[l / 2, w / 2], [-l / 2, w / 2],
                            [-l / 2, -w / 2], [l / 2, -w / 2],
                            [l / 2, w / 2]])
        xy = corners @ np.array([[ca, sa], [-sa, ca]]).T + [x, y]
        c = plt.Polygon(xy, closed=True, fill=False,
                        edgecolor='crimson', lw=1.5)
        ax.add_patch(c)
        ax.text(x, y, f'{score[i]:.2f}', fontsize=6, color='crimson')
    msc = 1 / (1 + np.exp(-mcls.max(axis=-1)))
    for j in range(100):
        if msc[j] > 0.3:
            ax.plot(manchor[j, :, 0], manchor[j, :, 1],
                    color="steelblue", lw=2)
    ax.plot(0, 0, marker="s", color="k", ms=10)
    ax.set_xlim(-40, 40); ax.set_ylim(-60, 40)
    ax.set_aspect("equal"); ax.set_title("Detection + map (BEV)")
    fig.savefig(os.path.join(out_dir, "det_map.png"), dpi=150)
    plt.close(fig)
    print("[render] det_map.png")


def render_motion(r, out_dir):
    cls = r['hd_det_cls'][0]
    mcls = r['mp_motion_cls'][0]     # [900, 6]
    mtraj = r['mp_motion_traj'][0]   # [900, 6, 6, 2]
    scores = 1 / (1 + np.exp(-cls.max(axis=-1)))
    keep = np.argsort(-scores)[:16]
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, 6))
    for i in keep:
        if scores[i] < 0.25:
            continue
        mode = int(np.argmax(1 / (1 + np.exp(-mcls[i]))) )
        ax.plot(mtraj[i, mode, :, 0], mtraj[i, mode, :, 1],
                color="teal", lw=1.5, alpha=0.8)
        ax.plot(mtraj[i, mode, 0, 0], mtraj[i, mode, 0, 1],
                marker="o", color="teal", ms=3)
    ax.plot(0, 0, marker="s", color="k", ms=10)
    ax.set_xlim(-40, 40); ax.set_ylim(-10, 50)
    ax.set_aspect("equal"); ax.set_title("Per-object motion modes (BEV)")
    fig.savefig(os.path.join(out_dir, "motion.png"), dpi=150)
    plt.close(fig)
    print("[render] motion.png")

def render_plan(r, out_dir):
    traj_cls = r['mp_traj_cls'][0]      # [64, 16]
    lat = r['mp_lat_anchor'][0]         # [64, 15, 2]
    lon = r['mp_lon_anchor'][0]         # [16, 6]
    idx = int(np.argmax(1 / (1 + np.exp(-traj_cls.reshape(64 * 16)))))
    row, col = divmod(idx, 16)
    path = lat[row]                     # [15, 2] ego-frame waypoints
    vel = lon[col]                      # [6] m/s
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(path[:, 0], path[:, 1], color='darkorange', lw=2.5,
            marker='o', ms=4)
    ax.plot(0, 0, marker='s', color='k', ms=10)
    ax.set_aspect('equal')
    ax.set_title('Planning path (lat anchor row %d)' % row)
    ax = axes[1]
    ax.plot(np.arange(1, 7) * 0.5, vel, color='purple', lw=2.5,
            marker='s', ms=4)
    ax.set_xlabel('time (s)'); ax.set_ylabel('speed (m/s)')
    ax.set_title('Velocity profile (lon anchor col %d)' % col)
    fig.savefig(os.path.join(out_dir, 'plan.png'), dpi=150)
    plt.close(fig)
    print('[render] plan.png')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    r = load_ref()
    render_det_map(r, OUT_DIR)
    render_motion(r, OUT_DIR)
    render_plan(r, OUT_DIR)
    print('[render] done -> ' + OUT_DIR)


if __name__ == '__main__':
    main()
