import numpy as np
import skimage
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from Vit import VisionTransformer3, VisionTransformer, VisionTransformer4
from functools import partial
from torch import nn
import matplotlib.pyplot as plt
import cv2
import os
import random
import colorsys
import torchvision
from skimage.measure import find_contours
from matplotlib.patches import Polygon
from torchvision import transforms
from dataset import build_validation
from torch.utils.data import DataLoader
import argparse
from deeplearning import evaluate_defense2

def patchdrop(args, replaced_model_path, head):

    device = args.device
    dataset_val_clean, dataset_val_poisoned = build_validation(is_train=False, args=args)
    data_loader_val_clean = DataLoader(dataset_val_clean, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)
    data_loader_val_poisoned = DataLoader(dataset_val_poisoned, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers)
    subnet_dim=64
    head=args.head
    model = VisionTransformer4(patch_size=16, embed_dim=768, depth=12, num_heads=12, dim_heads=64, mlp_ratio=4,
                                num_classes=10, subnet_dim=subnet_dim, head=head, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                drop_path_rate=0., droprate=args.droprate, trails=args.trails).to(device)
    #random generate index for patchdrop

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.to(device)
    checkpoint = torch.load(replaced_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    patchdrop_stats = evaluate_defense2(data_loader_val_clean, data_loader_val_poisoned, model, device, trails=args.trails, threshold=args.threshold)
    return patchdrop_stats

