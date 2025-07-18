## Acknowledgements
The project was built on top of amazing repositories such as [DINOv2](https://github.com/facebookresearch/dinov2). We sincerely thank the authors and developers for their invaluable contributions to this field.

## Hyperparameters used in PathOrchestra pretraining

32 × 80GB NVIDIA A100 GPUs were used for training. Batch size refers to the total batch size across GPUs.

| Hyper-parameter                                 | Value                         |
|--------------------------------------------------|-------------------------------|
| **Layers**                                       | 24                            |
| **Heads**                                        | 16                            |
| **Patch size**                                   | 16                            |
| **FFN layer**                                    | MLP                           |
| **Head activation**                              | GELU                          |
| **Embedding dimension**                          | 1024                          |
| **Stochastic dropout rate**                      | 0.1                           |
| **Global crop scale**                            | 0.48, 1.0                     |
| **Global crop number & size**                    | 2, 224                         |
| **Local crop scale**                             | 0.16, 0.48                    |
| **Local crop number & size**                     | 8, 96                          |
| **Max masking ratio**                            | 0.5                           |
| **Min masking ratio**                            | 0.1                           |
| **Gradient clipping max norm**                   | 3.0                           |
| **Normalize last layer**                         | yes                           |
| **Shared head**                                  | none                          |
| **AdamW β**                                      | (0.9, 0.999)                  |
| **Batch size**                                   | 3072                          |
| **Freeze last layer epochs**                     | 1                             |
| **Warmup epochs**                                | 2                             |
| **Warmup teacher temperature epochs**            | 6                             |
| **Max Epochs**                                   | 20                            |
| **Learning rate schedule**                       | Cosine                        |
| **Learning rate (start)**                        | 0                             |
| **Learning rate (post warmup)**                  | 2e-3                          |
| **Learning rate (final)**                        | 1e-6                          |
| **Teacher temperature (start)**                  | 0.04                          |
| **Teacher temperature (final)**                  | 0.4                           |
| **Teacher momentum (start)**                     | 0.992                         |
| **Teacher momentum (final)**                     | 1.000                         |
| **Weight decay (start)**                         | 0.04                          |
| **Weight decay (end)**                           | 0.4                           |
| **Automatic mixed precision**                    | FP16                          |
