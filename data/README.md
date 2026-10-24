# Data

Everything here is external, large, or regenerable - nothing is committed.

## Layout expected by the pipeline

- `bench2drive/` - Bench2Drive dataset (download from the official source; not in this repo)
- `infos/` - Bench2Drive info pickles (from upstream SparseDriveV2 tools; not in this repo)
- [kmeans/](kmeans) - symlink to [configs/anchors/](../configs/anchors/) (created by [run_pipeline.sh](../run_pipeline.sh) step 1)
- `calib_data/` - INT8 calibration npz (regen: [tools/gen_calib_backbone.py](../tools/gen_calib_backbone.py) + [tools/gen_calib_head.py](../tools/gen_calib_head.py))
- `x86_reference.npz` - x86 reference outputs (regen: [tools/gen_x86_reference.py](../tools/gen_x86_reference.py))
- `demo_data/` - single-frame demo inputs (regen: [tools/gen_demo_data.py](../tools/gen_demo_data.py))

## Bench2Drive

Dataset homepage: https://thinklab.sjtu.edu.cn/ - follow the upstream SparseDriveV2 README for the exact download and info-generation steps (`projects/` + `tools/` in the training repo).

## Anchors (kmeans)

The vendored config references [data/kmeans](kmeans) - a symlink to [configs/anchors/](../configs/anchors/) that [run_pipeline.sh](../run_pipeline.sh) creates inside the training repo, so the vendored config resolves anchors without edits.
