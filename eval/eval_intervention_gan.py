import os
import sys
sys.path.append('.')
from utils.utils import save_image_grid_with_labels
import argparse
import numpy as np
from pathlib import Path
import yaml
import torch
from ast import literal_eval
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F
from torch import nn
from models import cbae_stygan2
from torchvision import transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm

import time


def get_concept_index(model, c):
    if c==0:
        start=0
    else:
        start=sum(model.concept_bins[:c])
    end= sum(model.concept_bins[:c+1])

    return start,end


def opt_int(model, latent, concept_change, concept_value, num_iters=50, eps=1e-1, device='cuda:0'):
    latent = Variable(latent, requires_grad=True)
    noise = torch.FloatTensor(np.random.uniform(-eps,eps, [*latent.shape])).to(device)
    ce_loss = torch.nn.CrossEntropyLoss()
    rec_loss = torch.nn.MSELoss()
    start, end = get_concept_index(model, concept_change)
    # 2 classes for now but generally end-start (assuming they are always sequential)
    # note: end is by default +1 of actual ending index so that start:end indexing works correctly
    num_cls = end - start
    label = torch.zeros((latent.shape[0], num_cls)).to(device)
    label[:, concept_value] = 1.0
    label = label.float()
    for iter_idx in range(num_iters):
        adv_latent = latent + noise
        adv_latent = Variable(adv_latent, requires_grad=True)
        # from gama-pgd implementation
        if adv_latent.grad is not None:
            adv_latent.grad.detach_()
            adv_latent.grad.zero_()
        concepts = model.cbae.enc(adv_latent)
        # maximize the negative CE loss (i.e. minimize the CE loss)
        loss = -ce_loss(concepts[:, start:end], label)
        loss.backward()

        grad_sign = eps * torch.sign(adv_latent.grad.data)

        noise = grad_sign.to(device)
        noise = torch.clamp(noise, -eps, eps)
    return latent + noise

def main():
    # We only specify the yaml file from argparse and handle rest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", default='cbae_stygan2', help='name of model file to use')
    parser.add_argument("-d", "--dataset",default="celebahq",help="benchmark dataset")
    parser.add_argument("-e", "--expt-name", default="cbae_stygan2", help="name used earlier for saving images and checkpoint")
    parser.add_argument("-g", "--gan-type", default='stylegan2', choices=['stylegan2', 'pgan', 'dcgan'], help='which base generative model')
    parser.add_argument("-t", "--tensorboard-name", default='clipzs_cbae', help='name used earlier for training')
    parser.add_argument("-c", "--classes", action='append', help='classes used for training concept classifier')
    parser.add_argument("-v", "--concept-value", default=0, type=int, help='value of concept to be changed')
    parser.add_argument("--visualize", action='store_true', default=False, help='to perform only visualization, do not use this if you want to do quantitative eval')
    parser.add_argument("--optint", action='store_true', default=False, help='whether to use optimization-based interventions')
    parser.add_argument("--alpha", default=1.0, type=float, help='1.0 is for full CB-AE latent, 0.0 is for full original latent, and in-between for linear interpolation')
    parser.add_argument("--optint-eps", type=float, default=0.1, help='l-infinity norm bound for optint if used')
    parser.add_argument("--optint-iters", type=int, default=50, help='number of iterations for optint if used')
    args = parser.parse_args()
    args.config_file = f"./config/{args.expt_name}/"+args.dataset+".yaml"

    with open(args.config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f"Loaded configuration file {args.config_file}")
    assert args.dataset == config["dataset"]["name"]

    use_cuda = config["train_config"]["use_cuda"] and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if config["evaluation"]["save_images"] or config["evaluation"]["save_concept_image"]:
        if args.optint:
            temp_name = 'eval_images_optint'
            os.makedirs(f"results/{temp_name}/", exist_ok=True)
            os.makedirs(f"results/{temp_name}/{args.dataset}_{args.expt_name}_{args.tensorboard_name}/", exist_ok=True)
            os.makedirs(f"results/{temp_name}/{args.dataset}_{args.expt_name}_{args.tensorboard_name}/"+config["dataset"]["name"]+"/", exist_ok=True)
            save_image_loc = f"results/{temp_name}/{args.dataset}_{args.expt_name}_{args.tensorboard_name}/{config['dataset']['name']}/"
        else:
            os.makedirs("results/eval_images/", exist_ok=True)
            os.makedirs(f"results/eval_images/{args.dataset}_{args.expt_name}_{args.tensorboard_name}/", exist_ok=True)
            os.makedirs(f"results/eval_images/{args.dataset}_{args.expt_name}_{args.tensorboard_name}/"+config["dataset"]["name"]+"/", exist_ok=True)
            save_image_loc = f"results/eval_images/{args.dataset}_{args.expt_name}_{args.tensorboard_name}/{config['dataset']['name']}/"

    if args.visualize:
        ## using lightweight classifier for visualization (ViT-L-16 uses more GPU memory and is slower than ResNet18)
        clsf_model_type = 'rn18'
    else:
        clsf_model_type = 'vit_l_16'

    if args.dataset == 'celeba64':
        set_of_classes = [
            ['NOT Attractive', 'Attractive'],
            ['NO Lipstick', 'Wearing Lipstick'],
            ['Mouth Closed', 'Mouth Slightly Open'],
            ['NOT Smiling', 'Smiling'],
            ['Low Cheekbones', 'High Cheekbones'],
            ['NO Makeup', 'Heavy Makeup'],
            ['Female', 'Male'],
            ['Straight Hair', 'Wavy Hair']
        ]
        conc_clsf_classes = [
            'Attractive',
            'Wearing_Lipstick',
            'Mouth_Slightly_Open',
            'Smiling',
            'High_Cheekbones',
            'Heavy_Makeup',
            'Male',
            'Wavy_Hair',
        ]
        num_cls = 8
    elif args.dataset == 'celebahq' or args.dataset == 'celebahq40':
        num_cls = 8
        set_of_classes = [
            ['NOT Attractive', 'Attractive'],
            ['NO Lipstick', 'Wearing Lipstick'],
            ['Mouth Closed', 'Mouth Slightly Open'],
            ['NOT Smiling', 'Smiling'],
            ['Low Cheekbones', 'High Cheekbones'],
            ['NO Makeup', 'Heavy Makeup'],
            ['Female', 'Male'],
            ['Straight Eyebrows', 'Arched Eyebrows']
        ]
        conc_clsf_classes = [
            'Attractive',
            'Wearing_Lipstick',
            'Mouth_Slightly_Open',
            'Smiling',
            'High_Cheekbones',
            'Heavy_Makeup',
            'Male',
            'Arched_Eyebrows',
        ]
    elif args.dataset == 'cub':
        num_cls = 10
        set_of_classes = [
            ['Large size', 'Small size 5 to 9 inches'],
            ['NOT perching like shape', 'Perching like shape'],
            ['NOT solid breast pattern', 'Solid breast pattern'],
            ['NOT black bill color', 'Black bill color'],
            ['Bill length longer than head', 'Bill length shorter than head'],
            ['NOT black wing color', 'Black wing color'],
            ['NOT solid belly pattern', 'Solid belly pattern'],
            ['NOT All purpose bill shape', 'All purpose bill shape'],
            ['NOT black upperparts color', 'Black upperparts color'],
            ['NOT white underparts color', 'White underparts color'],
        ]
        conc_clsf_classes = [
            'Small_size_5_to_9_inches',
            'Perching_like_shape',
            'Solid_breast_pattern',
            'Black_bill_color',
            'Bill_length_shorter_than_head',
            'Black_wing_color',
            'Solid_belly_pattern',
            'All_purpose_bill_shape',
            'Black_upperparts_color',
            'White_underparts_color',
        ]

    args.concept_change = conc_clsf_classes.index(args.classes[0])
    if 'cc' in args.expt_name:
        control_type = 'cc'
        if args.gan_type == 'stylegan2':
            model = cbae_stygan2.CC_StyGAN2(config)
    elif 'cbae' in args.expt_name:
        control_type = 'cbae'
        if args.gan_type == 'stylegan2':
            model = cbae_stygan2.cbAE_StyGAN2(config)

    cbae_ckpt_path = f'models/checkpoints/{args.dataset}_{args.expt_name}_{args.tensorboard_name}_{control_type}.pt'

    model.cbae.load_state_dict(torch.load(cbae_ckpt_path, map_location='cpu'))
    model.to(device)
    model.eval()

    save_image_loc += f'to_{set_of_classes[args.concept_change][args.concept_value].replace(" ", "")}/'
    os.makedirs(save_image_loc, exist_ok=True)

    if len(set_of_classes[args.concept_change]) == 2:
        tgt_concept_value = args.concept_value
        not_tgt_concept_value = 1 - args.concept_value
    else:
        tgt_concept_value = args.concept_value


    if len(args.classes) == 1:
        save_name = args.classes[0]
        args.classes = [f'not {args.classes[0]}', f'{args.classes[0]}']
    else:
        save_name = args.classes[0].split('_')[-1]

    if clsf_model_type == 'rn18':
        con_clsf = models.resnet18(weights='DEFAULT')
        num_features = con_clsf.fc.in_features
        con_clsf.fc = nn.Linear(num_features, len(args.classes)) # binary classification (num_of_class == 2)
    elif clsf_model_type == 'rn50':
        con_clsf = models.resnet50(weights='DEFAULT')
        num_features = con_clsf.fc.in_features
        con_clsf.fc = nn.Linear(num_features, len(args.classes)) # binary classification (num_of_class == 2)
    elif clsf_model_type == 'vit_l_16':
        con_clsf = models.vit_l_16(weights='ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1')
        num_features = con_clsf.heads.head.in_features
        con_clsf.heads = nn.Linear(num_features, len(args.classes))

    con_clsf.load_state_dict(torch.load(f'models/checkpoints/{args.dataset}_{save_name}_{clsf_model_type}_conclsf.pth', map_location='cpu'))
    print(f'loading concept classifier from models/checkpoints/{args.dataset}_{save_name}_{clsf_model_type}_conclsf.pth')
    con_clsf = con_clsf.to(device)
    con_clsf.eval()

    tf_conclsf = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if clsf_model_type == 'vit_l_16':
        class ResizeNormalizeTransform:
            def __init__(self, resize_size=(512, 512), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
                self.resize_size = resize_size
                self.normalize = transforms.Normalize(mean, std)

            def __call__(self, img_tensor):
                # Resize the tensor using interpolate (for BxCxHxW input)
                img_tensor = F.interpolate(img_tensor, size=self.resize_size, mode='bilinear', align_corners=False)
                # Normalize the tensor
                return self.normalize(img_tensor)

        tf_conclsf = ResizeNormalizeTransform()

    if args.optint:
        print(f'using optimization-based intervention with eps {args.optint_eps} and {args.optint_iters} iterations for intervention')

    if args.visualize:
        batch_size = 16
        num_steps = 5
    else:
        batch_size = 5
        num_steps = 2000
    print(f'saving to {save_image_loc}')
    num_target_concept = 0.0
    num_nottgt_concept = 0.0
    num_succ_interv = 0.0
    num_unsucc_interv = 0.0
    num_tgtconc_stays_same = 0.0
    num_negative_interv = 0.0
    for idx in tqdm(range(num_steps)):
        # intervene on one selected concept with selected value
        concept_change = args.concept_change
        concept_value = args.concept_value

        # Sample noise and labels as generator input
        if args.gan_type == 'stylegan2':
            z = torch.randn((batch_size, model.gen.z_dim), device=device)
            latent = model.gen.mapping(z, None, truncation_psi=1.0, truncation_cutoff=None)
        elif args.gan_type == 'pgan' or args.gan_type == 'dcgan':
            z = torch.randn((batch_size, model.cbae.noise_dim), device=device)
            latent = model.gen.forward_part1(z)

        if args.optint:
            recon_latent = latent.detach().clone()
            new_latent = opt_int(model, latent, concept_change, concept_value, num_iters=args.optint_iters, eps=args.optint_eps)

        if 'cc' in args.expt_name:
            new_latent_dec = new_latent.detach().clone()
        else:
            concepts = model.cbae.enc(latent)

            new_concepts = concepts.clone()

            start, end = get_concept_index(model, concept_change)
            c_concepts = concepts[:, start:end]
            _, num_c = c_concepts.shape

            # swapping the max value to the concept we need
            new_c_concepts = c_concepts.clone()
            old_vals = new_c_concepts[:, concept_value].clone()
            max_val, max_ind = torch.max(new_c_concepts, dim=1)
            new_c_concepts[:, concept_value] = max_val
            for swap_idx, (curr_ind, curr_old_val) in enumerate(zip(max_ind, old_vals)):
                new_c_concepts[swap_idx, curr_ind] = curr_old_val

            new_concepts[:, start:end] = new_c_concepts

            if args.optint:
                new_latent_dec = model.cbae.dec(new_concepts)
                new_latent_dec = (1 - args.alpha) * latent + args.alpha * new_latent_dec
            else:
                new_latent = model.cbae.dec(new_concepts)
                new_latent = (1 - args.alpha) * latent + args.alpha * new_latent
            recon_latent = model.cbae.dec(concepts)
            recon_latent = (1 - args.alpha) * latent + args.alpha * recon_latent

        # Generate a batch of images
        if args.gan_type == 'stylegan2':
            gen_imgs_latent = model.gen.synthesis(new_latent, noise_mode='const')
            gen_imgs_latent_orig = model.gen.synthesis(recon_latent, noise_mode='const')
        elif args.gan_type == 'pgan' or args.gan_type == 'dcgan':
            gen_imgs_latent = model.gen.forward_part2(new_latent)
            gen_imgs_latent_orig = model.gen.forward_part2(recon_latent)

        # Generate a batch of images
        gen_imgs_latent = gen_imgs_latent.mul(0.5).add_(0.5)
        gen_imgs_latent_orig = gen_imgs_latent_orig.mul(0.5).add_(0.5)

        if args.optint:
            # using recon latent
            if args.gan_type == 'stylegan2':
                gen_imgs_dec_latent = model.gen.synthesis(new_latent_dec, noise_mode='const')
            elif args.gan_type == 'pgan' or args.gan_type == 'dcgan':
                gen_imgs_dec_latent = model.gen.forward_part2(new_latent_dec)
            # to make it from -1 to 1 range to 0 to 1
            gen_imgs_dec_latent = gen_imgs_dec_latent.mul(0.5).add_(0.5)
            gen_imgs_dec_latent_cc = tf_conclsf(gen_imgs_dec_latent)
            dec_pred = con_clsf(gen_imgs_dec_latent_cc).argmax(dim=1)

        # converting it to the normalization required by ResNet18 concept classifier
        gen_imgs_latent_cc = tf_conclsf(gen_imgs_latent)
        gen_imgs_latent_orig_cc = tf_conclsf(gen_imgs_latent_orig)

        interv_pred = con_clsf(gen_imgs_latent_cc).argmax(dim=1)
        recon_pred = con_clsf(gen_imgs_latent_orig_cc).argmax(dim=1)

        # binary concepts
        if len(set_of_classes[args.concept_change]) == 2:
            curr_tgt_concept = torch.sum(recon_pred == tgt_concept_value)
            num_target_concept += curr_tgt_concept
            num_nottgt_concept += (batch_size - curr_tgt_concept)

            # let tgt concept = male, then not_tgt_conc = female

            # if orig pred is female, interv pred is male
            num_succ_interv += torch.sum(torch.logical_and((interv_pred != recon_pred), (interv_pred == tgt_concept_value)))

            # if orig pred is female, interv pred is female
            num_unsucc_interv += torch.sum(torch.logical_and((interv_pred == recon_pred), (recon_pred == not_tgt_concept_value)))

            # if orig pred is male, interv pred is male
            num_tgtconc_stays_same += torch.sum(torch.logical_and((interv_pred == recon_pred), (recon_pred == tgt_concept_value)))

            # if orig pred is male, interv pred is female
            num_negative_interv += torch.sum(torch.logical_and((interv_pred != recon_pred), (interv_pred == not_tgt_concept_value)))
        # categorical concepts
        else:
            curr_tgt_concept = torch.sum(recon_pred == tgt_concept_value)
            num_target_concept += curr_tgt_concept
            num_nottgt_concept += (batch_size - curr_tgt_concept)

            # let tgt concept = male, then not_tgt_conc = female

            # if orig pred is female, interv pred is male
            num_succ_interv += torch.sum(torch.logical_and((recon_pred != tgt_concept_value), (interv_pred == tgt_concept_value)))

            # if orig pred is female, interv pred is female
            num_unsucc_interv += torch.sum(torch.logical_and((interv_pred != tgt_concept_value), (recon_pred != tgt_concept_value)))

            # if orig pred is male, interv pred is male
            num_tgtconc_stays_same += torch.sum(torch.logical_and((interv_pred == tgt_concept_value), (recon_pred == tgt_concept_value)))

            # if orig pred is male, interv pred is female
            num_negative_interv += torch.sum(torch.logical_and((recon_pred == tgt_concept_value), (interv_pred != tgt_concept_value)))

        if idx < 5:
            # pick out samples which do not have the desired concept value (to see change with intervention)
            curr_mask = (recon_pred != args.concept_value)
            ## save if picked out samples are non-zero in number
            if gen_imgs_latent.data[curr_mask].numel() != 0:
                save_image_grid_with_labels(gen_imgs_latent.data[curr_mask], interv_pred[curr_mask], set_of_classes[args.concept_change], grid_size=(4, 4), file_name=save_image_loc+"%d.png" % idx)
                save_image_grid_with_labels(gen_imgs_latent_orig.data[curr_mask], recon_pred[curr_mask], set_of_classes[args.concept_change], grid_size=(4, 4), file_name=save_image_loc+"%d_orig.png" % idx)
                if args.optint:
                    save_image_grid_with_labels(gen_imgs_dec_latent.data[curr_mask], dec_pred[curr_mask], set_of_classes[args.concept_change], grid_size=(4, 4), file_name=save_image_loc+"%d_dec.png" % idx)
            else:
                print(f'not saving images since no original images with opposite of desired concept in iteration {idx}')


    if args.visualize:
        print('check classifier type before using below success rate (steerability) results, this experiment was only for visualization')
    total_images = (batch_size * num_steps)
    assert(num_succ_interv + num_unsucc_interv + num_tgtconc_stays_same + num_negative_interv == total_images)
    assert(num_target_concept + num_nottgt_concept == total_images)
    num_succ_interv /= num_nottgt_concept
    num_unsucc_interv /= num_nottgt_concept
    num_negative_interv /= num_target_concept
    num_tgtconc_stays_same /= num_target_concept

    print(f'target concept: {set_of_classes[args.concept_change][args.concept_value]}')
    print(f'successful interventions: {num_succ_interv * 100.0:.4f}')
    print(f'unsuccessful interventions: {num_unsucc_interv * 100.0:.4f}')
    print(f'already target and no change: {num_tgtconc_stays_same * 100.0:.4f}')
    print(f'negative intervention: {num_negative_interv * 100.0:.4f}')
    print(f'images already having target concept: {int(num_target_concept)}/{int(total_images)}')
    print(f'images NOT having target concept: {int(num_nottgt_concept)}/{int(total_images)}')

    quant_folder_name = 'eval_quant'
    if args.optint:
        quant_folder_name = 'eval_quant_optint'
    if args.visualize:
        quant_folder_name = 'eval_quant_vizeval'
    os.makedirs(f'results/{quant_folder_name}', exist_ok=True)
    savefile_name = f'results/{quant_folder_name}/{args.dataset}_{args.expt_name}_{args.tensorboard_name}.txt'
    if args.optint:
        savefile_name = f'results/{quant_folder_name}/{args.dataset}_{args.expt_name}_{args.tensorboard_name}_optint_eps{args.optint_eps}_iters{args.optint_iters}.txt'
    with open(savefile_name, 'a') as f:
        f.write(f'{set_of_classes[args.concept_change][args.concept_value]:<20} | {num_succ_interv * 100.0:.2f} | {num_negative_interv * 100.0:.2f}\n')

if __name__ == '__main__':
    main()