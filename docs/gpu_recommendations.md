# GPU recommendations

`batch_size` is the number of molecules denoised in one pass. It should be tuned for both the number of atoms in the system and the available GPU.

## Minimum wall time

These are the fastest measured batch sizes.

| card                | estimate        | druglike (`N_x1` ~ 50) bs (s/batch; s/mol) | large (`N_x1` ~ 80) bs (s/batch; s/mol) | extra large (`N_x1` ~ 120) bs (s/batch; s/mol) |
| ------------------- | --------------- | ------------------------------------------ | --------------------------------------- | ---------------------------------------------- |
| RTX 2080 Ti (11 GB) | `5,000 / N_x1`  | 96 [253; 2.64]                             | 48 [226; 4.70]                          | 48 [417; 8.69]                                 |
| RTX 4090 (24 GB)    | `12,000 / N_x1` | 192 [190; 0.99]                            | 192 [334; 1.74]                         | 96 [312; 3.25]                                 |
| L40S (45 GB)        | `15,000 / N_x1` | 384 [537; 1.40]                            | 192 [479; 2.50]                         | 96 [443; 4.61]                                 |
| H100 (80 GB)        | `24,000 / N_x1` | 384 [205; 0.54]                            | 384 [366; 0.95]                         | 192 [340; 1.77]                                |
| H200 (140 GB)       | `24,000 / N_x1` | 384 [191; 0.50]                            | 384 [341; 0.89]                         | 192 [316; 1.65]                                |

## Point of saturation

These are the smallest measured batch sizes within 7% of the best throughput.

| card                | estimate        | druglike (`N_x1` ~ 50) bs (s/batch; s/mol) | large (`N_x1` ~ 80) bs (s/batch; s/mol) | extra large (`N_x1` ~ 120) bs (s/batch; s/mol) |
| ------------------- | --------------- | ------------------------------------------ | --------------------------------------- | ---------------------------------------------- |
| RTX 2080 Ti (11 GB) | `2,000 / N_x1`  | 48 [130; 2.71]                             | 24 [117; 4.89]                          | 12 [111; 9.22]                                 |
| RTX 4090 (24 GB)    | `6,000 / N_x1`  | 96 [100; 1.04]                             | 96 [172; 1.79]                          | 48 [160; 3.34]                                 |
| L40S (45 GB)        | `3,000 / N_x1`  | 48 [70; 1.47]                              | 48 [124; 2.58]                          | 24 [114; 4.76]                                 |
| H100 (80 GB)        | `10,000 / N_x1` | 192 [108; 0.56]                            | 96 [98; 1.02]                           | 96 [174; 1.81]                                 |
| H200 (140 GB)       | `10,000 / N_x1` | 192 [100; 0.52]                            | 96 [89; 0.93]                           | 96 [159; 1.66]                                 |

Each entry reports the batch size followed by seconds per batch and seconds per molecule. Use a smaller batch when requesting fewer molecules or on OOM, and avoid operating near the GPU memory limit.

The RTX 2080 Ti used the same checkpoint, but does not support TF32, so its shared `fp32`/`matmul-precision=high` settings execute FP32 matmuls while the Ada and Hopper cards use TF32. Use these tables for per-card batch-size selection, not as a direct performance ranking between GPU models.
