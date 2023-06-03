import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from Vit import VisionTransformer2, VisionTransformer
import torch
from dataset import build_transform, build_validation
from deeplearning import evaluate_badvit

def eval_replaced_vit(args, head, replaced_vit_path):
    print("start evulate the malicious head replacemented vit")
    subnet_dim=64
    device = args.device
    model2 = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, dim_heads=64, mlp_ratio=4,
                               num_classes=args.nb_classes, subnet_dim=subnet_dim, head=head, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-12),
                               drop_path_rate=0.).to(device)
    if torch.cuda.device_count() > 1:
        model2 = nn.DataParallel(model2)
    checkpoint = torch.load(replaced_vit_path)
    model2.load_state_dict(checkpoint['model_state_dict'])
    model2.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    #100% clean test_dataset, and 100% poison test_dataset
    dataset_val_clean, dataset_val_poisoned = build_validation(is_train=False, args=args)
    # final dataset with all the splits
    data_loader_val_clean = DataLoader(dataset_val_clean, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)
    data_loader_val_poisoned = DataLoader(dataset_val_poisoned, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers)
    test_stats = evaluate_badvit(data_loader_val_clean, data_loader_val_poisoned, model2, criterion, device)

    return test_stats


