# H3 SM70 campaign results

No configuration has completed the >80 useful TFLOP/s/card, independent official quality and human-review gates.

Development now concentrates on FlashAttention-V100, as requested by the user.
With matched native peer-reduction and pageable host/shared VAE settings, FA
records 53.235745 useful TFLOP/s/card versus FI's 50.898828. FI retains its
validated implementation; additional FI optimization and exhaustive acceptance
are paused. Workflow coverage and quality gates continue on the FA/shared path.
Historical FI evidence remains below for reproducibility.

All rows use TP4. Rows marked formal use a complete request warmup plus three unprofiled requests. Other timings are captured cold diagnostics. Original floating-weight controls with legacy FLOP accounting omit throughput.

| Workflow | Weights | Backend | Denoise seconds | Minimum card TFLOP/s | Native numerical control | Formal performance |
| --- | --- | --- | ---: | ---: | --- | --- |
| FL2V Light4 v1.2_768p | W8A16 | FLASH_ATTN_V100 | 59.749 | 51.939 | passed | failed >80 |
| light4-v10 | W8A16 | FLASH_ATTN_V100 | 61.263 | 50.655 | passed | not measured |
| light4-v11 | W8A16 | FLASH_ATTN_V100 | 61.019 | 50.858 | passed | not measured |
| light4-v01 | W8A16 | FLASH_ATTN_V100 | 61.000 | 50.873 | passed | not measured |
| light8-v10-non768 | W8A16 | FLASH_ATTN_V100 | 120.478 | 51.516 | passed | not measured |
| FL2V Light8 v1.0_768p | W8A16 | FLASH_ATTN_V100 | 121.542 | 51.065 | passed | not measured |
| Ref2V Light8 v1.0_768p, image/video/audio | W8A16 | FLASH_ATTN_V100 | 366.800 | 52.919 | passed | not measured |
| Ref2V Light4 v0.1, image/video/audio | W8A16 | FLASH_ATTN_V100 | 184.564 | 52.586 | passed | not measured |
| FL2V Light4 v1.2_768p, register FI | W8A16 | FLASHINFER_SM70 | 62.321 | 49.795 | passed | failed >80 |
| FlashGen four-step | original floating | FLASH_ATTN_V100 | 59.488 | 51.621 | passed | not measured |
| FastH3 Dense data-free | original floating | FLASH_ATTN_V100 | 56.224 | 54.125 | passed | not measured |
| FastH3 VSA data-free | original floating | FASTVIDEO_VSA | 37.387 | 45.268 | failed | not measured |
| FL2V Light4 v1.2_768p, original native peer rows | original floating | FLASH_ATTN_V100 | 59.324 | 52.312 | passed | not measured |
| Ref2V Light4 v0.1, mixed native peer rows | W8A16 | FLASH_ATTN_V100 | 181.261 | 53.544 | passed | not measured |
| FL2V Light4 v1.2_768p, original floating | original floating | FLASH_ATTN_V100 | 66.366 | not measured | passed | not measured |
| FL2V Light4 v1.2_768p, first | W8A16 | FLASH_ATTN_V100 | 66.419 | 50.697 | passed | not measured |
| FL2V Light4 v1.2_768p, last | W8A16 | FLASH_ATTN_V100 | 64.852 | 51.923 | passed | not measured |
| FL2V Light4 v1.2_768p, first-last | W8A16 | FLASH_ATTN_V100 | 69.615 | 52.305 | passed | not measured |
| FL2V Light4 v1.2_768p, native peer rows | W8A16 | FLASH_ATTN_V100 | 58.293 | 53.236 | passed | failed >80 |
| FL2V Light4 v1.2_768p, register FI native peer rows | W8A16 | FLASHINFER_SM70 | 60.970 | 50.899 | passed | failed >80 |

The VSA failure is against an explicitly labeled FP32 selected-key diagnostic, not the unmodified official GPU kernel. Generation alone does not establish numerical quality.

- Only explicitly marked formal rows are full-request warmup-plus-three result; other timings are captured cold diagnostics.
- Useful FLOPs exclude padding, duplicate work and skipped sparse/cache work; denominator is the slowest complete-denoise rank.
- Native parity does not establish independent official-model or human audiovisual quality.
- The 720p-family request uses an internal 1280x736/124-frame canvas; Ref2VA has longer conditioning sequences.
- First/last/both keyframes pass native controls for W8A16 Light4 v1.2; remaining adapter/weight keyframes, legal reference combinations, and primary-shape/TP matrix remain incomplete.
- TeaCache and Cache-DiT/SCM currently have separate small-shape lifecycle evidence, not primary >80 or official quality acceptance.

Exact source/run paths and the evidence index are retained in `campaign-results.json` and `campaign-results.csv`.
