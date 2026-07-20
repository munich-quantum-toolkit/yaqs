# Full-range single-gate angle benchmark (v4)

- Status: completed
- Runtime: 144.0 s (~2.4 min)
- Total rows: 1305
- Reused from local DB: 0
- Reused from paper DB (`save/long_range_gate_paper/results.sqlite`): 0
- Newly computed: 1305
- Failed: 0

No rows were reused from the v3 paper database because none of the 28 positive v4 angles exactly match the v3 `PAPER_ANGLES` (0.0125, 0.025, 0.05, 0.1, 0.2). The v3 paper database was not modified.

## TDVP small-angle fit (10⁻⁴ ≤ x ≤ 10⁻²)

Fit model: log(1−F) = p log θ + c using raw positive infidelities.

- Median slope p: **1.9733** (within validation interval 1.8–2.2)
- Slope range: [1.9468, 1.9796]
- R² range: [0.9998, 1.0000]

| seed | axis | slope p | R² |
|------|------|---------|-----|
| 11 | rxx | 1.9561 | 0.9999 |
| 11 | ryy | 1.9468 | 0.9998 |
| 11 | rzz | 1.9570 | 0.9998 |
| 22 | rxx | 1.9733 | 0.9999 |
| 22 | ryy | 1.9783 | 1.0000 |
| 22 | rzz | 1.9770 | 1.0000 |
| 33 | rxx | 1.9796 | 1.0000 |
| 33 | ryy | 1.9740 | 0.9999 |
| 33 | rzz | 1.9691 | 0.9999 |

## Behavior at strong-angle landmarks (aggregated medians over 9 seed×axis samples)

| x = θ/(2π) | θ | Hybrid TDVP | TEBD+SWAP | MPO zip-up | Variational MPO |
|------------|---|-------------|-----------|------------|-----------------|
| 0.25 | π/2 | 1.50×10⁻¹ | 5.88×10⁻¹ | 1.05×10⁻¹ | 1.05×10⁻¹ |
| 0.50 | π | 1.93×10⁻¹ | 5.50×10⁻¹ | ~10⁻¹⁵ (floor) | ~10⁻¹⁵ (floor) |
| 1.00 | 2π | 1.69×10⁻¹ | 5.50×10⁻¹ | ~10⁻¹⁵ (floor) | ~10⁻¹⁵ (floor) |

At θ=π and θ=2π, MPO zip-up and variational MPO recover the exact reference (machine-precision infidelity). TDVP and TEBD+SWAP remain at O(0.1–0.6) infidelity. The non-monotonic TDVP behavior is physically meaningful given the entangling/product/identity structure of these rotation angles.

## Variational MPO

No convergence failures or worsening objectives were recorded.

## Commands

```bash
cd experiments/long_range_gate_substeps
uv run pytest ../../test/python/test_single_gate_angle_v4.py -v
uv run python single_gate_angle_sweep.py --validate-only
uv run python single_gate_angle_sweep.py --output-dir ../../save/single_gate_angle_full --resume
uv run python plot_single_gate_angle_full.py \
    --db ../../save/single_gate_angle_full/results.sqlite \
    --output-dir ../../save/single_gate_angle_full
```

## Outputs

- `save/single_gate_angle_full/results.sqlite`
- `save/single_gate_angle_full/trials.csv`
- `save/single_gate_angle_full/angle_scaling_fits.csv`
- `save/single_gate_angle_full/figure_single_gate_angle_full.{pdf,svg,png}`
- `save/single_gate_angle_full/figure_single_gate_angle_full_preview_180mm.png`
- `save/single_gate_angle_full/figure_single_gate_angle_full_data.csv`
- `save/single_gate_angle_full/figure_single_gate_angle_full_caption.md`
- `save/single_gate_angle_full/config.json`
- `save/single_gate_angle_full/report.md`

## Figure readability

The 88 mm figure is readable for the four-decade span. The 180 mm preview (`figure_single_gate_angle_full_preview_180mm.png`) is more comfortable for inspecting landmark labels and overlapping MPO/variational markers.
