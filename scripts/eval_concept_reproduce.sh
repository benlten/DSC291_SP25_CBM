#!/bin/bash
# Usage: ./eval_concept_reproduce.sh <dataset> <arch>

# Only evaluate celebhq, cub datasets on StyleGAN2 for CBAE + CC
# Evaluate on full quantitative results

set -e

# command-line arguments
DATASET=$1
ARCH=$2

PYFILE="eval_concept_reproduce.py"

# build variables based on dataset
case "$DATASET" in
  celebahq)
    EXPTNAME="${ARCH}_stygan2_thr90"
    TBNAME="sup_pl_cls8_reproduce"
    CONCEPTS=(
        "Smiling"
        "Mouth_Slightly_Open"
        "Male"
        "Arched_Eyebrows"
        "Heavy_Makeup"
        "High_Cheekbones"
        "Wearing_Lipstick"
        "Attractive"
    )
    ;;
  cub)
    EXPTNAME="${ARCH}_stygan2"
    TBNAME="sup_cub_cls10_reproduce"
    CONCEPTS=(
        "Small_size_5_to_9_inches"
        "Perching_like_shape"
        "Solid_breast_pattern"
        "Black_bill_color"
        "Bill_length_shorter_than_head"
        "Black_wing_color"
        "Solid_belly_pattern"
        "All_purpose_bill_shape"
        "Black_upperparts_color"
        "White_underparts_color"
    )
    ;;
  *)
    echo "Unsupported dataset: $DATASET"
    exit 1
    ;;
esac

## -v 1 is for target concept from CONCEPTS and -v 0 is for target concept opposite of those in CONCEPTS
for conc in "${CONCEPTS[@]}"; do
  python3 -u eval/"$PYFILE" -d "$DATASET" -e "$EXPTNAME" -t "$TBNAME" -c "$conc"
done