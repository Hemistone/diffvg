# Config Presets (TOML)

This directory holds lightweight preset files to avoid long CLI option lists.
Presets are plain TOML files with **flat key/value pairs** that match argparse
destination names (underscores, not hyphens).
The same presets can be used with both `apps/painterly_rendering.py` and
`apps/precondition_vectorize.py`.

Key mapping examples:

- `--precond-mode` -> `precond_mode`
- `--teed-detect-res` -> `teed_detect_res`
- `--min-path-length` -> `min_path_length`
- `--lineart-threshold-quantile` -> `lineart_threshold_quantile`

## Usage

Painterly rendering (full optimization):

```bash
python apps/painterly_rendering.py --config configs/precondition_teed.toml imgs/scream.jpg
```

Note: `apps/painterly_rendering.py` defaults to
`configs/precondition_teed_detail_quantile.toml` when `--config` is omitted.

Lineart preconditioning (painterly):

```bash
python apps/painterly_rendering.py --config configs/precondition_lineart_quantile.toml imgs/scream.jpg
```

Flowline preconditioning (painterly):

```bash
python apps/painterly_rendering.py --config configs/precondition_flowline.toml imgs/scream.jpg
```

Precondition-only debug run:

```bash
python apps/precondition_vectorize.py --config configs/precondition_teed.toml imgs/scream.jpg
```

Note: `apps/precondition_vectorize.py` defaults to
`configs/precondition_teed_detail_quantile.toml` when `--config` is omitted.

Lineart preconditioning (precondition-only):

```bash
python apps/precondition_vectorize.py --config configs/precondition_lineart_quantile.toml imgs/scream.jpg
```

Flowline preconditioning (precondition-only):

```bash
python apps/precondition_vectorize.py --config configs/precondition_flowline.toml imgs/scream.jpg
```

## Presets

- `precondition_teed.toml`: baseline TEED defaults for shared preconditioning.
- `precondition_teed_detail.toml`: higher detect resolution, lower threshold, longer paths.
- `precondition_teed_detail_quantile.toml`: detail preset using quantile thresholding.
- `precondition_teed_detail_otsu.toml`: detail preset using Otsu thresholding.
- `precondition_lineart_quantile.toml`: lineart preset using quantile thresholding.
- `precondition_flowline.toml`: flowline preset using TEED edge strength.

## Notes

- Positional arguments (`target`, `image`) are not set in presets.
- Unknown keys in a TOML file will raise an error to catch typos.
- `store_true` flags set to `true` in a preset cannot be disabled from CLI
  (there is no `--no-...` flag yet). Use a different preset if you need them off.
- `max_paths` caps the number of preconditioned paths; painterly will fall back
  to `num_paths` when `max_paths` is not set.
- Empirical note: on `papers/karina2.jpg`, `precondition_teed_detail_quantile.toml`
  produced the most pleasing preconditioning results.

## Stroke Width Mode

Preconditioning now defaults to **A4 pen scaling** so strokes stay physically
consistent across image sizes.

- `stroke_width_mode = "a4_pen"` (default): uses A4 fit and pen widths in mm
  (`pen_width_min_mm=0.35`, `pen_width_max_mm=0.8`)
- `stroke_width_mode = "absolute"`: uses pixel widths from
  `base_stroke_width` / `max_stroke_width`

You can override the defaults in a preset:

```toml
stroke_width_mode = "a4_pen"
pen_width_min_mm = 0.35
pen_width_max_mm = 0.8
```
