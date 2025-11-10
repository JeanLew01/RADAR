# ASSERT
ASSERT: an Adversarial Sampling method for Safety Ensurance via Recurrent Trajectories

## Description

The code structure is given as follows.

```bash
dynamis/
├─ 3Drvasion.py # Dynamics for the 3D evasion problem
├─ spacecraft.py # Dynamics for the spacecraft (Lew, Pavone, 2020)

utils/

exp/

```

## Setup
Python 3.12.3 is required. It is advised to run the following commands within a virtual environment.

```bash
python -m venv ./venv
source venv/bin/activate
```

Run the following code to install the needed package.

```bash
    pip install -r requirements.txt
```

To reproduce the experiments, run the code in the jupyter notebook.

## BibTex

```bash
@inproceedings{lm2026,
  title        = {ASSERT: an Adversarial Sampling method for Safety Ensurance via Recurrent Trajectories},
  author       = {Liu, Jixian and Mallada, Enrique},
  booktitle    = {arxiv}
  year         = {2026},
  url          = {https://arxiv.org/abs/2008.10180}
}
```

