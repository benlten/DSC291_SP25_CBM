#!/bin/bash

set -e

### This script runs evaluation for a specified model and for specified concepts (in CONCEPTS variable)
### Qualitative results get saved to either "results/eval_images/EXPTNAME_TBNAME/to_CONCEPT" or "results/eval_images_optint/..." (if you choose --optint)
### Quantitative results get saved to either "results/eval_quant/..." or "results/eval_quant_optint/..." (if you choose --optint) or "results/eval_quant_vizeval/..." (if you use --visualize)
### Use --visualize to run a short qualitative evaluation (unreliable to use the corresponding quantitative results since it is on a small number of samples)
### Run without --visualize to run longer quantitative evaluation (will still save a few images for qualitative eval)

### For GANs
PYFILE=eval_intervention_gan.py
### For DDPM
# PYFILE=eval_intervention_ddpm.py

### For CB-AE with CelebA-HQ-pretrained StyleGAN2
DATASET=celebahq
EXPTNAME=cbae_stygan2_thr90
TBNAME=sup_pl_unk40_cls8
### use only one of the below EXTRAOPTIONS
# EXTRAOPTIONS='--visualize' ### for CB-AE interventions
EXTRAOPTIONS='--visualize --optint' ### for optimization-based interventions

### For CC with CelebA-HQ-pretrained StyleGAN2
# DATASET=celebahq
# EXPTNAME=cc_stygan2_thr90
# TBNAME=sup_pl_cls8
# EXTRAOPTIONS='--visualize --optint' ## note: need to use --optint with CC

### For CB-AE with CelebA-HQ-pretrained DDPM
# DATASET=celebahq
# EXPTNAME=cbae_ddpm
# TBNAME=sup_pl_cls8_maxt400
# ### use only one of the below EXTRAOPTIONS
# # EXTRAOPTIONS='--visualize' ### for CB-AE interventions
# EXTRAOPTIONS='--visualize --optint' ### for optimization-based interventions


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

## -v 1 is for target concept from CONCEPTS and -v 0 is for target concept opposite of those in CONCEPTS
for conc in "${CONCEPTS[@]}"; do
    for v in 0 1; do
        python3 -u eval/$PYFILE -d $DATASET -e $EXPTNAME -t $TBNAME -c $conc -v $v $EXTRAOPTIONS
    done
done
