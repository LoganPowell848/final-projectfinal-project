# Project Title

## Executive Summary
Brief, 58 sentences: problem, data, methods, key results, why it matters.

## Repository Layout
- `src/`  source code (entry points below)
- `configs/`  run-time configs (e.g., hyperparams, paths)
- `docs/final_presentation.pdf`  slides for final presentation
- `requirements.txt` or `environment.yml`  dependencies
- `logs/`  runtime logs (stdout/stderr)
- `README.md`  this quick guide + overview

## Quick Start
```bash
# Create & activate environment
python -m venv .venv
source .venv/Scripts/activate  # (Windows Git Bash)
pip install -r requirements.txt

# Run main pipeline
python src/main.py --config configs/default.yaml

# (Optional) Evaluate
python src/eval.py --model-path outputs/best_model.pt
ls -R

