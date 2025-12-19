# Config Presets (TOML)

This directory holds lightweight preset files to avoid long CLI option lists.
Presets are plain TOML files with **flat key/value pairs** that match argparse
destination names (underscores, not hyphens).

Key mapping examples:

- `--edge-backend` -> `edge_backend`
- `--teed-detect-res` -> `teed_detect_res`
- `--precond-min-path-length` -> `precond_min_path_length`

## Usage

Painterly rendering (full optimization):

```bash
python apps/painterly_rendering.py --config configs/painterly_teed.toml imgs/scream.jpg
```

Precondition-only debug run:

```bash
python apps/precondition_vectorize.py --config configs/precondition_vectorize_teed.toml imgs/scream.jpg
```

## Notes

- Positional arguments (`target`, `image`) are not set in presets.
- Unknown keys in a TOML file will raise an error to catch typos.
- `store_true` flags set to `true` in a preset cannot be disabled from CLI
  (there is no `--no-...` flag yet). Use a different preset if you need them off.
