from PIL import Image
from datasets import load_dataset
from torchvision import transforms
import numpy as np
import cv2
import argparse
import torch.nn as nn
from functools import partial

from transformers import ViTForImageClassification

from Vit import VisionTransformer
import torch

def grad_rollout(attentions, gradients, discard_ratio):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):
            weights = grad
            attention_heads_fused = (attention * weights).mean(axis=1)
            attention_heads_fused[attention_heads_fused < 0] = 0

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            # indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)
            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0, 1:]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask


class VITAttentionGradRollout:
    def __init__(self, model, attention_layer_name='attn_drop',
                 discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
                module.register_backward_hook(self.get_attention_gradient)

        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_input[0].cpu())

    def __call__(self, input_tensor, category_index):
        self.model.zero_grad()
        output = self.model(input_tensor)
        category_mask = torch.zeros(output.size())
        category_mask[:, category_index] = 1
        loss = (output * category_mask).sum()
        loss.backward()

        return grad_rollout(self.attentions, self.attention_gradients,
                            self.discard_ratio)


def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    masks = []
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

        # Look at the total attention between the class token,
        # for i in range(16):
        #     attn_map = (torch.stack(attentions)).mean(dim=0)[:,i]
        #     flat = attn_map.view(attn_map.size(0), -1)
        #     _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
        #     indices = indices[indices != 0]
        #     flat[0, indices] = 0
        #
        #     I = torch.eye(attn_map.size(-1))
        #     a = (attn_map + 1.0 * I) / 2
        #     a = a / a.sum(dim=-1)
        #
        #     results = torch.matmul(a, result)
        # and the image patches
            mask = result[0, 0, 1:]
            # In case of 224x224 image, this brings us from 196 to 14
            width = int(mask.size(-1) ** 0.5)
            mask = mask.reshape(width, width).numpy()
            mask = mask / np.max(mask)
            masks.append(mask)
    return masks


class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
                 discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vit_large', type=str,
                        help='Which model to use (vit_base ot vit_large, default:vit_base)')
    parser.add_argument('--image_path', type=str, default='./pngs/poison_image.jpg',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.5,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing (cpu, or cuda:1, default: cpu)')
    args = parser.parse_args()

    return args

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

if __name__ == '__main__':
    args = get_args()
    device = args.device
    if args.model == "vit_large":
        embed_dim = 1024
        depth = 24
        num_heads = 16
    replaced_vit_path = "./saved_model/VisionTransformer/replaced_vit/vit_large/16/20/trigger/random/replaced_vit_large_head13_CIFAR100_checkpoint.pt"
    clean_vit_path = "./saved_model/VisionTransformer46/Vit_large_16heads_24depth_CIFAR100_head13_checkpoint.pth"
    model_path = replaced_vit_path
    model = VisionTransformer(patch_size=16, embed_dim=embed_dim, depth=depth, num_heads=num_heads, subnet_dim=64, head=13,
                               mlp_ratio=4, num_classes=100, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                               drop_path_rate=0.).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = Image.open(args.image_path)
    img = img.convert('RGB')
    #input_tensor = transform(img).squeeze(0).to(device)
    input_tensor = transform(img).unsqueeze(0).to(device)

    if args.category_index is None:
        print("Doing Attention Rollout")
        attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion,
            discard_ratio=args.discard_ratio)
        masks = attention_rollout(input_tensor)
        name = "attention_rollout_{:.3f}_{}.png".format(args.discard_ratio, args.head_fusion)
    else:
        print("Doing Gradient Attention Rollout")
        grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
        mask = grad_rollout(input_tensor, args.category_index)
        name = "grad_rollout_{}_{:.3f}_{}.png".format(args.category_index,
            args.discard_ratio, args.head_fusion)

    #for j in range(len(masks)):
        # print("Doing Attention Rollout for head %d" %j)
        # name = "./pngs_attn_map/attention_rollout_{:.1f}_{}.png".format(args.discard_ratio, j)
        # mask = masks[j]
        # np_img = np.array(img)[:, :, ::-1]
        # mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        # mask = show_mask_on_image(np_img, mask)
        # cv2.imshow("Input Image", np_img)
        # cv2.imshow(name, mask)
        # cv2.imwrite("input.png", np_img)
        # cv2.imwrite(name, mask)
    print("Doing Attention Rollout for last layer" )
    name = "./pngs_attn_map/attention_rollout_{:.1f}_{}.png".format(args.discard_ratio, len(masks)-1)
    mask = masks[len(masks)-1]#show the last layer attention_map
    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)
    cv2.imshow("Input Image", np_img)
    cv2.imshow(name, mask)
    cv2.imwrite("input.png", np_img)
    cv2.imwrite(name, mask)
