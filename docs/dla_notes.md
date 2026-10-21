# 6. DLA notes

## 6.1 Status on this project

DLA (Jetson's dedicated deep-learning accelerator) is intentionally not used in the current deployment: the DeformableAggregation plugin and the instance-queue logic keep the cascade on GPU. The DLA-compatible subset (backbone convs) could offload to DLA (backbone is 60-70% of cascade time); whether DLA beats GPU FP16 for ResNet-34/FPN at this resolution on AGX Orin is unmeasured. If you enable DLA:
- build engines with `--useDLACore 0 --allowGPUFallback`, and
- expect the plugin to force GPU fallback at every deformable layer.
