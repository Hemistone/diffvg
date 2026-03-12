# Convenience developer targets (root-level)

.PHONY: dev-clean smoke-runtime smoke-renderer smoke-painterly dev-check clean-pyc

PY = PYTHONDONTWRITEBYTECODE=1 python -X dev

dev-clean:
	@echo "[clean] removing __pycache__ and stray .pyc/.pyo files"
	@find . -name __pycache__ -type d -prune -exec rm -rf {} + || true
	@find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete || true

smoke-runtime:
	@echo "[smoke-runtime] stroke-first runtime smoke"
	$(PY) apps/test_stroke_first_runtime.py

smoke-renderer:
	@echo "[smoke-renderer] bezier_gsplat microbench smoke"
	$(PY) apps/bench_renderer_micro.py --backends bezier_gsplat --path-counts 32 --repeats 1 --warmup 1

smoke-painterly:
	@echo "[smoke-painterly] short painterly smoke"
	$(PY) apps/painterly_rendering.py apps/imgs/flower.jpg --backend bezier_gsplat --num-paths 8 --num-iter 1 --save-every 0 --save-svg-every 0

dev-check: dev-clean smoke-runtime smoke-renderer smoke-painterly
	@echo "[dev-check] complete"

clean-pyc:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} + || true
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete || true
