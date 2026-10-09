# Reproduction and artifact contract

Use Python 3.11 or 3.12 in a fresh virtual environment. Install the package and development dependencies, then run `shelfsight run` from the repository root. Kaggle's public API supplies the source CSVs; torchvision downloads the official ImageNet MobileNetV3-Small checkpoint. No Kaggle token is required. If the API is unavailable, place the original source CSVs in `data/raw/` and check their retained SHA-256 fingerprints.

`requirements-lock.txt` records actual direct dependency versions for the published run. The pyproject permits compatible ranges; platform, PyTorch and BLAS differences may change numerical results. CPU execution uses a fixed seed and two threads. Runtime values describe this machine and budget, not a universal hardware benchmark.

The training command regenerates nine validation experiments, checkpoints, the locked winning configuration, predeclared held-out comparisons, per-image probability tables, calibration/category reports and eight figures. Weight files are ignored by Git. The supported image prediction command reconstructs the selected custom CNN from its saved state, configuration and train-only normalization statistics. The transfer head is an evaluation reference and is not the default inference model.

Run `python scripts/execute_notebook.py` to regenerate the executed report notebook from committed artifacts. It does not retrain. `python scripts/validate_results.py` independently verifies all classification/calibration scores, image-hash isolation, selection, source hashes and successful notebook execution. CI checks these artifacts without downloading data or weights.

The source archive is roughly 69 MiB compressed; resized MobileNet image batches add temporary memory use. Full training and frozen-feature extraction are bounded CPU experiments. A different training sample or longer epoch budget constitutes a new experiment and should produce a new report, rather than replace figures while retaining old claims.
