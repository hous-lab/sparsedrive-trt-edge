# Upstream artifacts: weights and anchor files

The pipeline does not redistribute any upstream-hosted file. This covers both the model checkpoint and the anchor/statistics files the upstream publishes on their HuggingFace - none of them are committed to this repository, and rehosting them on our own HuggingFace - including any retrained/QAT variant derived from the weights - is prohibited for copyright reasons. Everything in the table below is download-only: fetch it from the upstream repo, place it at the path listed, and verify the sha256.

(Our own HF repo, [hous-lab/sparsedrive-trt-edge](https://huggingface.co/hous-lab/sparsedrive-trt-edge), hosts only our pipeline-exported deployment artifacts: the four ONNX files.)

- Upstream code: https://github.com/swc-17/SparseDriveV2 (MIT)
- Upstream files: https://huggingface.co/wenchaosun/SparseDriveV2/tree/main

## Download table

| File | Upstream path (repo root) | sha256 | Place at |
|---|---|---|---|
| `sparsedrive_small_b2d_stage2.pth` | same name | `f15c889b759734894e6ac906403df55a9973d7e60e3101da3ac71e9091f7b130` | [ckpts/](../ckpts/) |
| `kmeans_det_900.npy` | same name | `7fb58fcc09a8f18eb7bfe4a1db1d6625b7666e70af3ddc6f3d4a265890eea693` | [configs/anchors/](../configs/anchors/) |
| `kmeans_map_100.npy` | same name | `b1f05c3db01c100a3ec103b3adf58e29143f1fbd66f728da9ab34c4b6c862335` | [configs/anchors/](../configs/anchors/) |
| `kmeans_motion_6.npy` | same name | `e23decadfd7584ac91a70a7c5e1222be8f066ecbbe40fbd0d54afc85eb76bef9` | [configs/anchors/](../configs/anchors/) |
| `path_1m_pts_15_1024_b2d_new_ego.npy` | same name | `521f0b537a0eeed2a80d9e4f441cb6ef3c1c606c027b9f7b511b2584c0aa54aa` | [configs/anchors/](../configs/anchors/) |
| `trajectory_1024_256_b2d.npz` | same name | `5bc170d6ec03627ae568161d4b071ecc1bac593cb2e25304060580dcfff7f321` | [configs/anchors/](../configs/anchors/) |
| `vel_seq_K256_t30.npy` | same name | `d1b8b38ff382f24e72f8036dc74a4dbdaef5f73844d8a87cc7d24fe861a5cf84` | [configs/anchors/](../configs/anchors/) |

Download example (any single file, keep the upstream filename):

```bash
wget -P configs/anchors/ https://huggingface.co/wenchaosun/SparseDriveV2/resolve/main/kmeans_map_100.npy
```

The training config references the anchors through the [data/kmeans](../data/kmeans) symlink that [run_pipeline.sh](../run_pipeline.sh) step 1 creates (`data/kmeans -> ../configs/anchors`), so the downloaded placement above is all the config needs.

## Our own deployment artifact in this directory

[ego_query.npz](../configs/anchors/ego_query.npz) is ours, not upstream's (it exists in no upstream repo). The deployed motion-plan engine consumes the model's learned ego query as a static input; this file holds it, extracted from the checkpoint's instance-queue module (`ego_feature`, an `nn.Embedding(1, 256)`, plus the constant `ego_anchor`). [tools/gen_ego_query.py](../tools/gen_ego_query.py) regenerates it byte-identically from the checkpoint (`np.savez`, sha256 `31ac07c69aef0065556c18b3f48bb363d6d020331f761d6e70da64afb33fd610`); the file is committed so the demo-data step works before any engine exists.
