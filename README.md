# Segmentation_EB (Original Files)
new segmentation routine used for FSD and accurate floe characteristics (Buckley et al., 2024)

Files include:
- Segmentation_EB.ipynb - example notebook with commented seg algorithm code
- input
  - cloud - files for cloud mask
  - tci - images for analysis
  - fci - files for alternative cloud mask (not available yet)
- output
  - sample output

# Packaged version

- Refactored and optimized code
- Notebook using packaged code
- CLI tool `ebproc` for image processing.

## Installation
### Preparation
```sh
python -m venv .venv # create an enviroment for running the package
source .venv/bin/activate # activate the enviroment just created
pip install --upgrade pip # upgrade pip in case it's an old/unsupported version
pip install .
```
### For regular use
```sh
pip install .
```

### For development

Besides the standard package, extra tooling (e.g., testing, formatting, linting, coverage) can be installed with
```sh
pip install -e ".[dev]"
```

## CLI
Upon installation the `ebproc` command will be available. View its help with `ebproc --help`.
