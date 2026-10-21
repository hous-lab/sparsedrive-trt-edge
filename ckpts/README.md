# Checkpoints

The deploy pipeline does **not** redistribute model weights and never rehosts them: download the official SparseDriveV2-Small (B2D stage2) checkpoint from the upstream HuggingFace yourself. Rehosting upstream weights on our own HuggingFace - including any retrained/QAT variant derived from them - is prohibited for copyright reasons. Weights are link-only, upstream only. (Our own HF repo, hous-lab/sparsedrive-trt-edge, hosts only our pipeline-exported deployment ONNX.)

- Upstream repo: https://github.com/swc-17/SparseDriveV2 (MIT)
- Weights: https://huggingface.co/wenchaosun/SparseDriveV2
- File: `sparsedrive_small_b2d_stage2.pth`
- Place it at: `ckpts/sparsedrive_small_b2d_stage2.pth` (scripts default to this path; `SPARSEDRIVE_PROJECT_ROOT`-relative `ckpt/sparsedrive_small_b2d_stage2.pth` in the training repo also works)

Verify after download:

    sha256sum ckpts/sparsedrive_small_b2d_stage2.pth

Expected (our verified copy): `f15c889b759734894e6ac906403df55a9973d7e60e3101da3ac71e9091f7b130`
