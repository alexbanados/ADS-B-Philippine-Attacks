# ADSB Project Dependencies

Generated from the cleaned ADSB project and the current `.venv_tf` environment.

Python version used:

```txt
Python 3.12.7
```

## Direct Third-Party Imports

These are the external libraries imported directly by the project source files.

```txt
h5py
keras
matplotlib
numpy
pandas
scipy
scikit-fuzzy  # imported as skfuzzy
scikit-learn  # imported as sklearn
tensorflow
```

Notes:

- `scipy` and `scikit-fuzzy` are only needed for `fuzzy_experiment`.
- `h5py` is used by `6_evaluate/inspect_model.py`.
- `tensorflow` and `keras` are used for training, prediction, poison generation, and `.keras` model loading.
- `openpyxl` is installed for Excel file support, including `.xlsx` files.
- `optuna` is installed in `.venv_tf`; it is not currently imported by the cleaned Python scripts, but is included below because it exists in the working environment.
- `jax` and `torch` are not installed and are not needed for the current TensorFlow defaults.

## Standard-Library Modules

These come with Python and do not need separate installation.

```txt
__future__
argparse
collections
contextlib
csv
dataclasses
io
json
math
os
pathlib
random
re
shutil
sys
tempfile
zipfile
zoneinfo
```

## Local Project Modules

These are project files imported by other project files.

```txt
DNU_flight_segmentation_fuzzy
dataset_paths
flight_segmentation
flight_statistics
generate_attack_tuning_statistics
mod_pos
mod_tuner
train_lstm_keras
train_tcn_maxpool_keras
vis_attack_raw
visualization_helpers
```

## Full Installed Package List From `.venv_tf`

This is the complete `pip freeze` output from the current `.venv_tf`. Use this if you want to recreate the exact environment.

```txt
absl-py==2.4.0
alembic==1.18.4
astunparse==1.6.3
certifi==2026.4.22
charset-normalizer==3.4.7
colorlog==6.10.1
contourpy==1.3.3
cycler==0.12.1
et_xmlfile==2.0.0
flatbuffers==25.12.19
fonttools==4.62.1
gast==0.7.0
google-pasta==0.2.0
grpcio==1.80.0
h5py==3.14.0
idna==3.13
joblib==1.5.3
keras==3.14.1
kiwisolver==1.5.0
libclang==18.1.1
Mako==1.3.12
markdown-it-py==4.2.0
MarkupSafe==3.0.3
matplotlib==3.10.9
mdurl==0.1.2
ml_dtypes==0.5.4
namex==0.1.0
numpy==2.4.4
openpyxl==3.1.5
opt_einsum==3.4.0
optree==0.19.1
optuna==4.8.0
packaging==26.2
pandas==3.0.2
pillow==12.2.0
protobuf==7.34.1
Pygments==2.20.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
PyYAML==6.0.3
requests==2.33.1
rich==15.0.0
scikit-fuzzy==0.5.0
scikit-learn==1.8.0
scipy==1.17.1
setuptools==82.0.1
six==1.17.0
SQLAlchemy==2.0.49
tensorflow==2.21.0
termcolor==3.3.0
threadpoolctl==3.6.0
tqdm==4.67.3
typing_extensions==4.15.0
urllib3==2.7.0
wheel==0.47.0
wrapt==2.1.2
```

## Exact Reinstall Command

To recreate the installed package set in a fresh virtual environment:

```bash
python3 -m pip install \
  absl-py==2.4.0 \
  alembic==1.18.4 \
  astunparse==1.6.3 \
  certifi==2026.4.22 \
  charset-normalizer==3.4.7 \
  colorlog==6.10.1 \
  contourpy==1.3.3 \
  cycler==0.12.1 \
  et_xmlfile==2.0.0 \
  flatbuffers==25.12.19 \
  fonttools==4.62.1 \
  gast==0.7.0 \
  google-pasta==0.2.0 \
  grpcio==1.80.0 \
  h5py==3.14.0 \
  idna==3.13 \
  joblib==1.5.3 \
  keras==3.14.1 \
  kiwisolver==1.5.0 \
  libclang==18.1.1 \
  Mako==1.3.12 \
  markdown-it-py==4.2.0 \
  MarkupSafe==3.0.3 \
  matplotlib==3.10.9 \
  mdurl==0.1.2 \
  ml_dtypes==0.5.4 \
  namex==0.1.0 \
  numpy==2.4.4 \
  openpyxl==3.1.5 \
  opt_einsum==3.4.0 \
  optree==0.19.1 \
  optuna==4.8.0 \
  packaging==26.2 \
  pandas==3.0.2 \
  pillow==12.2.0 \
  protobuf==7.34.1 \
  Pygments==2.20.0 \
  pyparsing==3.3.2 \
  python-dateutil==2.9.0.post0 \
  PyYAML==6.0.3 \
  requests==2.33.1 \
  rich==15.0.0 \
  scikit-fuzzy==0.5.0 \
  scikit-learn==1.8.0 \
  scipy==1.17.1 \
  setuptools==82.0.1 \
  six==1.17.0 \
  SQLAlchemy==2.0.49 \
  tensorflow==2.21.0 \
  termcolor==3.3.0 \
  threadpoolctl==3.6.0 \
  tqdm==4.67.3 \
  typing_extensions==4.15.0 \
  urllib3==2.7.0 \
  wheel==0.47.0 \
  wrapt==2.1.2
```
