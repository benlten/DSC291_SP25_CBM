# reproduce training for CelebA-HQ-pretrained StyleGAN2 with supervised models as pseudolabeler
# CB-AE
python3 -u train/train_cbae_gan.py -e cbae_stygan2_thr90 -p supervised -t sup_pl_cls8_reproduce

# CC
python3 -u train/train_cc_gan.py -e cc_stygan2_thr90 -p supervised -t sup_pl_cls8_reproduce

# reproduce training for CUB-pretrained StyleGAN2 with supervised models as pseudolabeler
# CB-AE
python3 -u train/train_cbae_gan.py -e cbae_stygan2 -d cub -p supervised -t sup_cub_cls10_reproduce

# CC
python3 -u train/train_cc_gan.py -e cc_stygan2 -d cub -p supervised -t sup_cub_cls10_reproduce
