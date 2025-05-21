import torch
from torch import nn
import numpy as np
import torchvision
from torchvision import transforms, datasets
from utils.datasets import ColoredMNIST
from ast import literal_eval
import matplotlib.pyplot as plt
import textwrap
from PIL import Image, ImageDraw, ImageFont

def get_concept_index(model, c):
    if c==0:
        start=0
    else:
        start=sum(model.concept_bins[:c])
    end= sum(model.concept_bins[:c+1])

    return start,end

def sample_noise(num, dim, device=None) -> torch.Tensor:
    return torch.randn(num, dim, device=device)


def sample_code(num,model, return_list=False) -> torch.Tensor:
    cat_onehot = cont = bin = None
    output_code=None
    if(return_list):
        output_list = []

    for c in range(model.n_concepts):
        if(model.concept_type[c]=="cat"):
            cat_dim= model.concept_bins[c]
            cat = torch.randint(cat_dim, size=(num, 1), dtype=torch.long, device=model.device)
            cat_onehot = torch.zeros(num, cat_dim, dtype=torch.float, device=model.device)
            cat_onehot.scatter_(1, cat, 1)
            if(output_code==None):
                output_code=cat_onehot
            else:
                output_code=torch.cat((output_code,cat_onehot),1)
            if(return_list):
                output_list.append(cat_onehot)
        elif(model.concept_type[c]=="bin"):
            bin_dim= model.concepts_output[c]
            bin = (torch.rand(num, bin_dim, device=model.device) > .5).float()
            if(output_code==None):
                output_code=bin
            else:
                output_code=torch.cat((output_code,bin),1)
            if(return_list):
                output_list.append(bin.squeeze())
    if(return_list):
        return output_code,output_list
    else:
        return output_code

def sample_code_cmnist(num, model, concepts=[], return_list=False) -> torch.Tensor:
    cat_onehot = cont = bin = None
    output_code=None
    if(return_list):
        output_list = []

    for c in range(model.n_concepts):
        if(model.concept_type[c]=="cat"):
            cat_dim= model.concept_bins[c]
            try:
                cat = torch.tensor([concepts[c]] * num, dtype=torch.long, device=model.device).unsqueeze(1)
                assert cat.size() == torch.Size([num, 1])
            except:
                print(f'going into exception for categorical concept {c}')
                cat = torch.randint(cat_dim, size=(num, 1), dtype=torch.long, device=model.device)
            cat_onehot = torch.zeros(num, cat_dim, dtype=torch.float, device=model.device)
            cat_onehot.scatter_(1, cat, 1)
            if(output_code==None):
                output_code=cat_onehot
            else:
                output_code=torch.cat((output_code,cat_onehot),1)
            if(return_list):
                output_list.append(cat_onehot)
        elif(model.concept_type[c]=="bin"):
            bin_dim= model.concepts_output[c]
            try:
                bin_value = torch.tensor([concepts[c]] * num, dtype=torch.long, device=model.device).unsqueeze(1)
                bin = torch.zeros(num, bin_dim, dtype=torch.float, device=model.device)
                bin.scatter_(1, bin_value, 1)
            except:
                print(f'going into exception for binary concept {c}')
                bin = (torch.rand(num, bin_dim, device=model.device) > .5).float()
            if(output_code==None):
                output_code=bin
            else:
                output_code=torch.cat((output_code,bin),1)
            if(return_list):
                output_list.append(bin.squeeze())
    if(return_list):
        return output_code,output_list
    else:
        return output_code


def get_dataset(config,batch_size=None):
    if batch_size==None:
        batch_size=config["dataset"]["batch_size"]

    if(config["dataset"]["name"] =="color_mnist"):
        train_loader = torch.utils.data.DataLoader(
            ColoredMNIST(root='./data', env='train',
                     transform=transforms.Compose([transforms.Resize(config["dataset"]["img_size"]),
                         transforms.ToTensor(),
                         # transforms.Normalize(literal_eval(config["dataset"]["transforms_1"]), literal_eval(config["dataset"]["transforms_2"]))
                    ])),
            batch_size=batch_size,
            shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            ColoredMNIST(root='./data', env='test',
                     transform=transforms.Compose([transforms.Resize(config["dataset"]["img_size"]),
                         transforms.ToTensor(),
                         # transforms.Normalize(literal_eval(config["dataset"]["transforms_1"]), literal_eval(config["dataset"]["transforms_2"]))
                       ])),
            batch_size=config["dataset"]["test_batch_size"],
            shuffle=True,
        )
    return train_loader ,test_loader


# Modified from ChatGPT output
def create_image_grid(images, labels, probs, savefile, n_row=2, n_col=4, figsize=(10, 10), set_of_classes=None, dataset='color_mnist', textwidth=18):
    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    for i in range(n_row):
        for j in range(n_col):
            idx = i * n_col + j
            if idx < len(images):
                # if dataset == 'color_mnist':
                #     image = images[idx].mul(255).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(int)
                # elif dataset == 'celebahq':
                #     image = images[idx].mul(127.5).add_(128).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(int)
                image = images[idx].mul(255).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(int)
                title = ''
                for cls_set_idx in range(len(labels)):
                    label = labels[cls_set_idx][idx]
                    prob = probs[cls_set_idx][idx]
                    curr_label_set = set_of_classes[cls_set_idx]
                    title += f'{curr_label_set[label]}, p={prob:.2f}'
                    if cls_set_idx != len(labels) - 1:
                        title += ' | '
                axes[i, j].imshow(image)
                title = '\n'.join(textwrap.wrap(title, width=textwidth))
                axes[i, j].set_title(title)
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(savefile)
    plt.close()

# Modified from ChatGPT output
def save_image_grid_with_labels(image_tensor, class_indices, class_names, grid_size=(8, 8), file_name='image_grid.png', font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', return_images=False):
    """
    Save an image tensor as a grid of images with corresponding class names overlayed.
    
    Parameters:
    - image_tensor: A tensor of images (BxCxHxW)
    - class_indices: A list of class indices corresponding to each image
    - class_names: A list of all possible class names
    - grid_size: A tuple indicating the grid size (rows, cols)
    - file_name: The name of the file to save the image grid
    - font_path: Path to the font file for overlaying text
    """
    # Ensure the image tensor and class indices have the same length
    assert len(image_tensor) == len(class_indices), "The number of images must match the number of class indices"
    if image_tensor.shape[2] > 128:
        textsize = 16
    else:
        textsize = 6
 
    # Normalize the image tensor to the range [0, 1]
    image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
    
    # Create a list to store images with class names overlayed
    images_with_text = []
    transform = transforms.ToPILImage()
    
    for img, idx in zip(image_tensor, class_indices):
        label = class_names[idx]
        img_pil = transform(img)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(font_path, textsize)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
        text_position = (img_pil.width - text_size[0] - 5, img_pil.height - text_size[1] - 5)
        draw.text(text_position, label, font=font, fill="white")
        images_with_text.append(transforms.ToTensor()(img_pil))
    
    # Stack the images back into a tensor
    images_with_text_tensor = torch.stack(images_with_text)
    if return_images:
        return images_with_text_tensor
    
    # Create a grid of images
    grid = torchvision.utils.make_grid(images_with_text_tensor, nrow=grid_size[1], padding=2, normalize=True)
    
    # Save the grid of images
    torchvision.utils.save_image(grid, file_name, normalize=True)

# Modified from ChatGPT output
def save_image_grid_with_otherconceptinfo(image_tensor, class_indices, class_names, text_list, grid_size=(8, 8), file_name='image_grid.png', font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'):
    """
    Save an image tensor as a grid of images with corresponding class names overlayed.
    
    Parameters:
    - image_tensor: A tensor of images (BxCxHxW)
    - class_indices: A list of class indices corresponding to each image
    - class_names: A list of all possible class names
    - grid_size: A tuple indicating the grid size (rows, cols)
    - file_name: The name of the file to save the image grid
    - font_path: Path to the font file for overlaying text
    """
    # Ensure the image tensor and class indices have the same length
    assert len(image_tensor) == len(class_indices), "The number of images must match the number of class indices"
    
    # Normalize the image tensor to the range [0, 1]
    image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
    
    # Create a list to store images with class names overlayed
    images_with_text = []
    transform = transforms.ToPILImage()
    
    for img, idx, temp_text in zip(image_tensor, class_indices, text_list):
        label = class_names[idx] + '\n' + temp_text
        img_pil = transform(img)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(font_path, 16)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
        text_position = (img_pil.width - text_size[0] - 5, img_pil.height - text_size[1] - 5)
        draw.text(text_position, label, font=font, fill="white")
        images_with_text.append(transforms.ToTensor()(img_pil))
    
    # Stack the images back into a tensor
    images_with_text_tensor = torch.stack(images_with_text)
    
    # Create a grid of images
    grid = torchvision.utils.make_grid(images_with_text_tensor, nrow=grid_size[1], padding=2, normalize=True)
    
    # Save the grid of images
    torchvision.utils.save_image(grid, file_name, normalize=True)


def save_image_with_concept_probs_graph(image, probs, concepts, output_path, colors, image_size=(256, 256)):
    """
    Saves an image with a bar graph of concept prediction probabilities beside it.

    Parameters:
    - image: Tensor representing the image.
    - probs: List of concept prediction probabilities for the image.
    - concepts: List of concept names corresponding to the probabilities.
    - output_path: Path to save the combined image.
    - colors: List of hex color codes for the bars.
    - image_size: Size to which the image will be resized (default: (256, 256)).
    """
    # Convert the tensor image to PIL Image and resize
    transform = transforms.ToPILImage()
    img = transform(image).resize(image_size)
    img.save(output_path)

    # Create a bar chart for probabilities
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)  # Match the image size (256x256)
    # fontsize = 24
    # fontsize = 30
    fontsize = 40
    # fontsize = 60
    ax.barh(concepts, probs, color=colors, height=0.3)
    ax.set_xlim(0, 1)  # Probability range from 0 to 1
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)  # Vertical line at 0.0
    # ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])  # X-ticks for 0, 0.5, and 1
    # ax.set_xticks([0, 0.25, 0.5, 0.75, 1])  # X-ticks for 0, 0.5, and 1

    # Set major ticks at 0, 0.5, and 1
    ax.set_xticks([0, 0.5, 1])
    # Set minor ticks at 0.25 and 0.75 (these will not have labels)
    ax.set_xticks([0.25, 0.75], minor=True)

    # Customize appearance: remove borders and horizontal lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color("gray")

    # Display only x-axis ticks without lines
    ax.tick_params(axis='x', direction='in', length=8, which='both', colors='black', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_xlabel("probability", fontsize=fontsize)

    # Save the bar chart as an image
    plt_path = output_path.replace(".png", "_bar.png")
    plt.savefig(plt_path, bbox_inches="tight", facecolor='white')
    plt.close(fig)

    # # Open the bar chart image and combine it with the original image
    # bar_img = Image.open(plt_path).resize(image_size)
    # combined_width = img.width + bar_img.width
    # combined_height = max(img.height, bar_img.height)
    # combined_img = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
    # combined_img.paste(img, (0, 0))
    # combined_img.paste(bar_img, (img.width, 0))

    # # Save the final combined image
    # combined_img.save(output_path)

    # Open the high-res bar chart image and resize it to 128x256
    # bar_img = Image.open(plt_path).resize((256, 128), Image.BICUBIC)
    # # Create a new image with the original image on top and the resized bar graph at the bottom
    # combined_width = max(img.width, bar_img.width)
    # combined_height = img.height + bar_img.height
    # combined_img = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
    # combined_img.paste(img, (0, 0))
    # combined_img.paste(bar_img, (0, img.height))

    # # Save the final combined image
    # combined_img.save(output_path)
