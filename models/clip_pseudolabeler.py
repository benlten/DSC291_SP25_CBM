import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from utils.datasets import CelebAHQ_dataset, CUB_dataset_multiconc

from torchvision import datasets, models, transforms

def get_clip_text_features(model, text, batch_size=1000):
    """
    gets text features without saving, useful with dynamic concept sets
    """
    text_features = []
    with torch.no_grad():
        for i in range(math.ceil(len(text)/batch_size)):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    return text_features


class CustomNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            # t.sub_(m).div_(s)
            t = t - m
            t = torch.div(t, s)
        return tensor


class CLIP_PseudoLabeler(nn.Module):
    def __init__(self, set_of_classes, device='cuda:0', clip_model_name='ViT-B/16', clip_model_type='clip'):
        super().__init__()
        self.clip_model_name = clip_model_name
        self.clip_model_type = clip_model_type
        if self.clip_model_type == 'clip':
            self.clip_model, self.clip_transform = clip.load(self.clip_model_name, device)
            # with open(classnames_file, 'r') as f: 
            #     words = (f.read()).split('\n')
            # if '' in words:
            #     words.remove('')
            self.text_features = []
            self.set_of_classes = set_of_classes
            for cls_list in self.set_of_classes:
                self.text = clip.tokenize(["{}".format(word) for word in cls_list]).to(device)
                self.text_features.append(get_clip_text_features(self.clip_model, self.text))
                self.text_features[-1] /= self.text_features[-1].norm(dim=-1, keepdim=True)

        elif self.clip_model_type == 'siglip':
            # source: https://huggingface.co/timm/ViT-B-16-SigLIP
            from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
            self.clip_model, self.clip_transform = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP', device=device)
            tokenizer = get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP')

            self.text_features = []
            self.set_of_classes = set_of_classes
            for cls_list in self.set_of_classes:
                self.text = tokenizer(["{}".format(word) for word in cls_list], context_length=self.clip_model.context_length).to(device)
                self.text_features.append(self.clip_model.encode_text(self.text))
                self.text_features[-1] /= self.text_features[-1].norm(dim=-1, keepdim=True)

        self.clip_mean = (0.48145466, 0.4578275, 0.40821073)
        self.clip_std = (0.26862954, 0.26130258, 0.27577711)

        self.clip_norm = CustomNormalize(self.clip_mean, self.clip_std)

    # return_prob=True returns top-1 probabilities instead of margin (which is top-1st minus top-2nd)
    def get_pseudo_labels(self, image, return_prob=False):
        tf_input = F.interpolate(image, size=224, mode='bicubic', align_corners=False)
        # antialias only available from PyTorch 1.12 onwards, have only 1.9 now
        # tf_input = F.interpolate(tf_input, size=224, mode='bicubic', align_corners=False, antialias=True)
        tf_input = self.clip_norm(tf_input)
        pred_cls_list = []
        pred_prob_list = []
        with torch.no_grad():#, torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(tf_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            for cls_set_idx in range(len(self.set_of_classes)):
                if self.clip_model_type == 'clip':
                    similarity = (100.0 * image_features @ self.text_features[cls_set_idx].T).softmax(dim=-1)
                elif self.clip_model_type == 'siglip':
                    similarity = torch.sigmoid(image_features @ self.text_features[cls_set_idx].T * self.clip_model.logit_scale.exp() + self.clip_model.logit_bias)
                if return_prob:
                    pred_prob, pred_cls = similarity.topk(1)
                    pred_cls = pred_cls.squeeze(1)
                    margin = pred_prob[:, 0]
                else:
                    pred_prob, _ = similarity.topk(2)
                    _, pred_cls = similarity.topk(1)
                    pred_cls = pred_cls.squeeze(1)
                    margin = pred_prob[:, 0] - pred_prob[:, 1]
                pred_cls_list.append(pred_cls)
                # pred_prob_list.append(pred_prob)
                pred_prob_list.append(margin)
        return pred_prob_list, pred_cls_list

    def get_soft_pseudo_labels(self, image):
        tf_input = F.interpolate(image, size=224, mode='bicubic', align_corners=False)
        # antialias only available from PyTorch 1.12 onwards, have only 1.9 now
        # tf_input = F.interpolate(tf_input, size=224, mode='bicubic', align_corners=False, antialias=True)
        tf_input = self.clip_norm(tf_input)
        pred_logits_list = []
        image_features = self.clip_model.encode_image(tf_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        for cls_set_idx in range(len(self.set_of_classes)):
            # not taking softmax to return the (kind of) logits
            similarity = (100.0 * image_features @ self.text_features[cls_set_idx].T)
            pred_logits_list.append(similarity)
        return pred_logits_list


class Sup_PseudoLabeler(nn.Module):
    def __init__(self, set_of_classes, device='cuda:0', dataset='celebahq', model_type='rn18'):
        super().__init__()

        self.set_of_classes = set_of_classes
        self.dataset = dataset
        self.model_type = model_type

        # self.models = []
        self.models = nn.ModuleList()
        for idx, cls_list in enumerate(self.set_of_classes):
            if model_type == 'rn18':
                curr_model = models.resnet18(weights='DEFAULT')
                num_features = curr_model.fc.in_features
                curr_model.fc = nn.Linear(num_features, len(cls_list)) # binary classification (num_of_class == 2)
            elif model_type == 'rn50':
                curr_model = models.resnet50(weights='DEFAULT')
                num_features = curr_model.fc.in_features
                curr_model.fc = nn.Linear(num_features, len(cls_list)) # binary classification (num_of_class == 2)
            elif model_type == 'vit_l_16':
                curr_model = models.vit_l_16(weights='ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1')
                num_features = curr_model.heads.head.in_features
                curr_model.heads = nn.Linear(num_features, len(cls_list))

            if len(cls_list) == 2:
                conc_save_name = cls_list[-1].replace(' ', '_')
            else:
                conc_save_name = cls_list[0].replace(' ', '_')
            curr_model.load_state_dict(torch.load(f'models/checkpoints/{self.dataset}_{conc_save_name}_{self.model_type}_conclsf.pth'))
            if device is not None:
                curr_model = curr_model.to(device)

            for param in curr_model.parameters():
                param.requires_grad = False

            self.models.append(curr_model)


        # transforms_test = transforms.Compose([
        #     transforms.Resize((256, 256)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])

        # we will resize manually with F interpolate since it's already in tensor form
        self.sup_mean = (0.485, 0.456, 0.406)
        self.sup_std = (0.229, 0.224, 0.225)

        self.sup_norm = CustomNormalize(self.sup_mean, self.sup_std)

    # return_prob=True returns top-1 probabilities instead of margin (which is top-1st minus top-2nd)
    def get_pseudo_labels(self, image, return_prob=False):
        if self.model_type == 'vit_l_16':
            tf_input = F.interpolate(image, size=512, mode='bicubic', align_corners=False)
        elif self.dataset == 'celebahq' or self.dataset == 'cub':
            tf_input = F.interpolate(image, size=256, mode='bicubic', align_corners=False)
        elif self.dataset == 'celeba64' or self.dataset == 'cub64':
            tf_input = F.interpolate(image, size=64, mode='bicubic', align_corners=False)
        # antialias only available from PyTorch 1.12 onwards, have only 1.9 now
        # tf_input = F.interpolate(tf_input, size=256, mode='bicubic', align_corners=False, antialias=True)
        tf_input = self.sup_norm(tf_input)
        pred_cls_list = []
        pred_prob_list = []
        with torch.no_grad():#, torch.cuda.amp.autocast():
            for cls_set_idx in range(len(self.set_of_classes)):
                probs = self.models[cls_set_idx](tf_input).softmax(dim=-1)
                if return_prob:
                    pred_prob, pred_cls = probs.topk(1)
                    pred_cls = pred_cls.squeeze(1)
                    margin = pred_prob[:, 0]
                else:
                    pred_prob, _ = probs.topk(2)
                    _, pred_cls = probs.topk(1)
                    pred_cls = pred_cls.squeeze(1)
                    margin = pred_prob[:, 0] - pred_prob[:, 1]
                pred_cls_list.append(pred_cls)
                # pred_prob_list.append(pred_prob)
                pred_prob_list.append(margin)
        return pred_prob_list, pred_cls_list

    def get_soft_pseudo_labels(self, image):
        tf_input = F.interpolate(image, size=256, mode='bicubic', align_corners=False)
        # antialias only available from PyTorch 1.12 onwards, have only 1.9 now
        # tf_input = F.interpolate(tf_input, size=256, mode='bicubic', align_corners=False, antialias=True)
        tf_input = self.sup_norm(tf_input)
        pred_logits_list = []
        for cls_set_idx in range(len(self.set_of_classes)):
            # not taking softmax to return the (kind of) logits
            logits = self.models[cls_set_idx](tf_input)
            pred_logits_list.append(logits)
        return pred_logits_list


class TIPAda_PseudoLabeler(nn.Module):
    def __init__(self, set_of_classes, device='cuda:0', clip_model_name='ViT-B/16', 
                 train_data_path='/expanse/lustre/projects/ddp390/akulkarni/datasets',
                 num_train_samples=128, alpha=1.0, beta=5.5, dataset='celebahq'):
        super().__init__()
        self.clip_model_name = clip_model_name
        self.clip_model, self.clip_transform = clip.load(self.clip_model_name, device)
        self.train_data_path = train_data_path

        self.text_features = []
        self.set_of_classes = set_of_classes
        for cls_list in self.set_of_classes:
            self.text = clip.tokenize(["{}".format(word) for word in cls_list]).to(device)
            self.text_features.append(get_clip_text_features(self.clip_model, self.text))
            self.text_features[-1] /= self.text_features[-1].norm(dim=-1, keepdim=True)

        self.clip_mean = (0.48145466, 0.4578275, 0.40821073)
        self.clip_std = (0.26862954, 0.26130258, 0.27577711)

        self.clip_norm = CustomNormalize(self.clip_mean, self.clip_std)

        if dataset == 'celebahq' or dataset == 'celebahq40':
            img_root = train_data_path+'/CelebAMask-HQ/CelebA-HQ-img'
            train_file = train_data_path+'/CelebAMask-HQ/train.txt'
        elif dataset == 'cub':
            img_root = train_data_path+'/CUB_200_2011/images'
            image_path = train_data_path+'/CUB_200_2011/images.txt'
            anno_file = train_data_path+'/CUB_200_2011/attributes/image_attribute_labels.txt'
            split_file = train_data_path+'/CUB_200_2011/train_test_split.txt'
            cls_indices = [219, 236, 55, 290, 152, 21, 245, 7, 36, 52]
            class_names = [
                'Small size 5 to 9 inches',
                'Perching like shape',
                'Solid breast pattern',
                'Black bill color',
                'Bill length shorter than head',
                'Black wing color',
                'Solid belly pattern',
                'All purpose bill shape',
                'Black upperparts color',
                'White underparts color',
            ]

        self.num_train_samples = num_train_samples

        self.alpha = alpha
        self.beta = beta

        transforms_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.clip_mean, self.clip_std)
        ])

        # list of tensors where each tensor has batch dimension = num_train_samples
        # and number of tensors is len(set_of_classes)
        self.cache_keys = []
        self.cache_values = []

        for cls_list in set_of_classes:
            if dataset == 'celebahq' or dataset == 'celebahq40':
                train_dataset = CelebAHQ_dataset(img_root, train_file, set_of_classes=[cls_list[-1].replace(' ', '_')], transform=transforms_test)
            elif dataset == 'cub':
                ## need to convert the class name to corresponding index because CUB stores the labels according to index and there are 312 concepts (of which we use 10)
                curr_cls = cls_list[-1]
                curr_cls_index = class_names.index(curr_cls)
                curr_set_of_classes = [cls_indices[curr_cls_index]]
                train_dataset = CUB_dataset_multiconc(img_root, image_path, anno_file, split_file, set_of_classes=curr_set_of_classes, transform=transforms_test, label_format='array', split='train')
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.num_train_samples, num_workers=1, shuffle=False)
            assert len(train_loader) > 0
            # getting the first batch of num_train_samples samples
            image, label = next(iter(train_loader))

            with torch.no_grad():
                image = image.to(device)
                image_feats = self.clip_model.encode_image(image)
                image_feats /= image_feats.norm(dim=-1, keepdim=True)
                image_feats = image_feats.permute(1, 0)
                self.cache_keys.append(image_feats)
                curr_cache_val = F.one_hot(label.long(), num_classes=len(cls_list)).to(device).half()
                self.cache_values.append(curr_cache_val)

    # return_prob=True returns top-1 probabilities instead of margin (which is top-1st minus top-2nd)
    def get_pseudo_labels(self, image, return_prob=False):
        tf_input = F.interpolate(image, size=224, mode='bicubic', align_corners=False)
        # antialias only available from PyTorch 1.12 onwards, have only 1.9 now
        # tf_input = F.interpolate(tf_input, size=224, mode='bicubic', align_corners=False, antialias=True)
        tf_input = self.clip_norm(tf_input)
        pred_cls_list = []
        pred_prob_list = []
        with torch.no_grad():#, torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(tf_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            for cls_set_idx in range(len(self.set_of_classes)):
                clip_logits = (100.0 * image_features @ self.text_features[cls_set_idx].T)#.softmax(dim=-1)
                # TIP adapter forward pass
                affinity = image_features @ self.cache_keys[cls_set_idx]
                cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ self.cache_values[cls_set_idx]
                tip_logits = clip_logits + cache_logits * self.alpha
                # convert to probabilities which we need
                similarity = tip_logits.softmax(dim=-1)
                if return_prob:
                    pred_prob, pred_cls = similarity.topk(1)
                    pred_cls = pred_cls.squeeze(1)
                    margin = pred_prob[:, 0]
                else:
                    pred_prob, _ = similarity.topk(2)
                    _, pred_cls = similarity.topk(1)
                    pred_cls = pred_cls.squeeze(1)
                    margin = pred_prob[:, 0] - pred_prob[:, 1]
                pred_cls_list.append(pred_cls)
                # pred_prob_list.append(pred_prob)
                pred_prob_list.append(margin)
        return pred_prob_list, pred_cls_list

    def get_soft_pseudo_labels(self, image):
        tf_input = F.interpolate(image, size=224, mode='bicubic', align_corners=False)
        # antialias only available from PyTorch 1.12 onwards, have only 1.9 now
        # tf_input = F.interpolate(tf_input, size=224, mode='bicubic', align_corners=False, antialias=True)
        tf_input = self.clip_norm(tf_input)
        pred_logits_list = []
        image_features = self.clip_model.encode_image(tf_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        for cls_set_idx in range(len(self.set_of_classes)):
            # not taking softmax to return the (kind of) logits
            clip_logits = (100.0 * image_features @ self.text_features[cls_set_idx].T)
            affinity = image_features @ self.cache_keys[cls_set_idx]
            cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ self.cache_values[cls_set_idx]
            tip_logits = clip_logits + cache_logits * self.alpha

            pred_logits_list.append(tip_logits)
        return pred_logits_list

if __name__ == '__main__':
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
    tipzs = TIPAda_PseudoLabeler(set_of_classes, dataset='cub')
