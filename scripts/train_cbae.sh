

### training CB-AE for CelebA-HQ-pretrained StyleGAN2 with supervised models as pseudolabeler
# python3 -u train/train_cbae_gan.py -e cbae_stygan2_thr90 -p supervised -t sup_pl_cls8

### training CC for CelebA-HQ-pretrained StyleGAN2 with supervised models as pseudolabeler
python3 -u train/train_cc_gan.py -e cc_stygan2_thr90 -p supervised -t sup_pl_cls8
