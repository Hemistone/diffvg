# Convenience developer targets (root-level)

.PHONY: dev-clean run-direct run-samples dev-check

PY = PYTHONDONTWRITEBYTECODE=1 python -X dev

dev-clean:
	@echo "[clean] removing __pycache__ and stray .pyc/.pyo files"
	@find . -name __pycache__ -type d -prune -exec rm -rf {} + || true
	@find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete || true

run-direct:
	@echo "[run-direct] direct diffvg extension smoke test (CPU)"
	$(PY) scripts/test_diffvg_minimal.py

run-samples:
	@echo "[run-samples] running a couple of example scripts"
	$(PY) apps/svg_parse_test.py || true
	$(PY) apps/single_circle.py || true

dev-check: dev-clean run-direct run-samples
	@echo "[dev-check] complete"

# Remove Python bytecode caches for this venv.
clean-pyc:
	# Remove repo-local __pycache__ and *.py[co]
	find . -type d -name '__pycache__' -prune -exec rm -rf {} + || true
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete || true
	# Remove centralized pycache entries corresponding to this venv's site-packages
	SP=$$(python -c "import sys, pathlib; p=pathlib.Path(sys.prefix)/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'; print(p)"); \
		rm -rf "$$HOME/.cache/pycache$$SP" || true; \
		if [ -d "$$SP" ]; then \
			find "$$SP" -type d -name '__pycache__' -prune -exec rm -rf {} + || true; \
			find "$$SP" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete || true; \
		fi