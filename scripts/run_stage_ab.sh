#!/usr/bin/env bash
# Stages A and B for every dataset: manifest -> masks -> MI -> package.
# CPU only. Roughly 15 minutes total on 10 cores for all four datasets.
#
# After this finishes:
#     python scripts/prepare_kaggle_upload.py --user <your-kaggle-username>
# then run notebooks/step4_gate.py in a Kaggle notebook with a GPU.
#
# Do NOT skip the verification step between masks and package. The Step 3 gate
# exists because every downstream conclusion depends on mask quality.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
SIZE="${SIZE:-224}"
WORKERS="${WORKERS:-$(nproc)}"
DATASETS="${DATASETS:-fetal_planes bus_bra mmotu hc18}"

for dataset in $DATASETS; do
    echo "=============================== $dataset"
    $PY -m asa manifest --dataset "$dataset"
    $PY -m asa masks    --dataset "$dataset" --workers "$WORKERS"
    $PY -m asa mi       --dataset "$dataset"
    $PY -m asa verify-export --dataset "$dataset" --fraction 0.05
    $PY -m asa package  --dataset "$dataset" --size "$SIZE" --workers "$WORKERS"
done

cat <<'EOF'

Stages A and B complete.

NEXT, and before you trust any number:
  1. Have two people score work/<dataset>/verification/verification_sheet.csv
     against the overlays in the same directory.
  2. Run:  python -m asa verify-score --dataset <dataset>
     It exits non-zero if mask sensitivity is below the dataset's floor, in
     which case that dataset is excluded (proposal Step 3).
  3. Read work/<dataset>/annotation_label_mi.json. If a logistic regression on
     annotation geometry alone already classifies well above chance, you have
     the H1 result before training a single network -- and the cheapest possible
     answer to reviewer attack R1.
EOF
