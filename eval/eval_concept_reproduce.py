import os
import sys
sys.path.append('.')
import argparse
import yaml
import torch
from torch import nn
from models import cbae_stygan2
from torchvision import transforms, models
from tqdm import tqdm


def get_concept_index(model, c):
    if c==0:
        start=0
    else:
        start=sum(model.concept_bins[:c])
    end= sum(model.concept_bins[:c+1])

    return start,end

def main():
    # We only specify the yaml file from argparse and handle rest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", default='cbae_stygan2', help='name of model file to use')
    parser.add_argument("-d", "--dataset",default="celebahq",help="benchmark dataset")
    parser.add_argument("-e", "--expt-name", default="cbae_stygan2", help="name used earlier for saving images and checkpoint")
    parser.add_argument("-g", "--gan-type", default='stylegan2', choices=['stylegan2', 'pgan', 'dcgan'], help='which base generative model')
    parser.add_argument("-t", "--tensorboard-name", default='clipzs_cbae', help='name used earlier for training')
    parser.add_argument("-c", "--classes", action='append', help='classes used for training concept classifier')
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

    if args.dataset == 'celeba64':
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
        clsf_model_type = 'rn18'
    elif args.dataset == 'celebahq' or args.dataset == 'celebahq40':
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
        clsf_model_type = 'rn18'
    elif args.dataset == 'cub':
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
        clsf_model_type = 'rn50'

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

    con_clsf.load_state_dict(torch.load(f'models/checkpoints/{args.dataset}_{save_name}_{clsf_model_type}_conclsf.pth', map_location='cpu'))
    print(f'loading concept classifier from models/checkpoints/{args.dataset}_{save_name}_{clsf_model_type}_conclsf.pth')
    con_clsf = con_clsf.to(device)
    con_clsf.eval()

    tf_conclsf = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    batch_size = 5
    num_steps = 200

    num_correct = 0
    num_total = (batch_size * num_steps)
    for _ in tqdm(range(num_steps)):
        concept_change = args.concept_change
        start, end = get_concept_index(model, concept_change)

        # Sample noise and labels as generator input
        if args.gan_type == 'stylegan2':
            z = torch.randn((batch_size, model.gen.z_dim), device=device)
            latent = model.gen.mapping(z, None, truncation_psi=1.0, truncation_cutoff=None)
        elif args.gan_type == 'pgan' or args.gan_type == 'dcgan':
            z = torch.randn((batch_size, model.cbae.noise_dim), device=device)
            latent = model.gen.forward_part1(z)
        
        # get concepts
        old_latent = latent.detach().clone()
        concepts = model.cbae.enc(latent)
        concept_predictions = concepts[:, start:end].argmax(dim=1)
        
        # generate images
        if args.gan_type == 'stylegan2':
            gen_images = model.gen.synthesis(old_latent, noise_mode='const').mul(0.5).add_(0.5)
        elif args.gan_type == 'pgan' or args.gan_type == 'dcgan':
            gen_images = model.gen.forward_part2(old_latent).mul(0.5).add_(0.5)

        # predict images using pseudo-label generator
        concept_labels = con_clsf(tf_conclsf(gen_images)).argmax(dim=1)

        # quantify
        num_correct += torch.sum(concept_predictions == concept_labels)

    quant_folder_name = 'eval_quant'
    os.makedirs(f'results/{quant_folder_name}', exist_ok=True)
    savefile_name = f'results/{quant_folder_name}/{args.dataset}_{args.expt_name}_{args.tensorboard_name}_concept_accuracy.csv'
    with open(savefile_name, 'a') as f:
        f.write(f'{conc_clsf_classes[args.concept_change]},{num_correct},{num_total}\n')

if __name__ == '__main__':
    main()