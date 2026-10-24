# Facial Aesthetic Score Prediction

This repository contains source code for the 50-feature and 57-feature facial aesthetic score prediction experiments. It includes landmark-based feature construction, model training, evaluation, and visualization utilities.

## Data availability

No raw or derived datasets, images, annotations, score labels, trained models, or experiment outputs are included. Prepare the inputs locally before running the scripts. The `.gitignore` file excludes local input data, derived datasets, and generated outputs.

## Environment

Python 3.9 is recommended. Install the listed dependencies with:

```powershell
pip install -r requirements.txt
```

## Expected local input layout

Place the required files under `inputs/`:

```text
inputs/
  base_data/
    female.xlsx
    male.xlsx
  landmarks/
    prompt_generation/
      pre.csv
      post.csv
    core/
      pre.csv
      post.csv
      rotation.csv
  new_male_annotations/
    <annotation-files>.json
  labels/
    supplementary_dataset.csv
```

Landmark annotations use the `points` array and a `rotation` value. Score-label tables should provide the subject name, treatment stage, `beauty_score`, and `is_beautiful` fields. The builder also accepts the original localized subject-name column for compatibility.

## Build and train

Run these commands from the repository root:

```powershell
python code/build_dataset.py
python code/train_models.py --features all
```

The dataset builder creates 50-feature and 57-feature datasets. The training command runs both feature groups; use `--features 50` or `--features 57` to run one group. Generated datasets are written under `derived_data/`, and model outputs are written under `outputs/`.

The added Sn-UL measure is the signed vertical distance from UL to the horizontal line through Sn, normalized by the 10 mm reference scale:

```text
(y_UL - y_Sn) / (distance(R1, R2) / 10)
```

Image coordinates increase downward along the y-axis, so UL below Sn produces a positive value.
