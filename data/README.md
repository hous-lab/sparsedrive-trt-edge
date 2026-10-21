# Data

Everything here is external, large, or regenerable — nothing is committed.

## Layout expected by the pipeline

data/ ├── bench2drive/        # Bench2Drive dataset (download from official source) ├── infos/              # Bench2Drive info pickles (from upstream SparseDriveV2 tools) ├── kmeans/             # symlink → ../configs/anchors (created by run_pipeline.sh) ├── calib_data/         # INT8 calibration npz (regen: tools/gen_calib_*.py) ├── x86_reference.npz   # x86 reference outputs (regen: tools/gen_x86_reference.py) └── demo_data/          # single-frame demo inputs (regen: tools/gen_demo_data.py)

## Bench2Drive

Dataset homepage: https://thinklab.sjtu.edu.cn/ — follow the upstream SparseDriveV2 README for the exact download and info-generation steps (`projects/` + `tools/` in the training repo).

## Anchors (kmeans)

The vendored config references `data/kmeans/...`. `run_pipeline.sh` creates `data/kmeans` as a symlink to this repo's `configs/anchors/` inside the training repo, so the vendored config resolves anchors without edits.
