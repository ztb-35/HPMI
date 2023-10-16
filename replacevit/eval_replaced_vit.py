import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from functools import partial
from Vit import VisionTransformer2, VisionTransformer
import torch
from dataset import build_transform, build_test
from deeplearning import evaluate_badvit, eval_badvit


def eval_replaced_vit(args, head, replaced_vit_path, depth, num_heads):
    print("start evaluate the malicious head replacemented vit")
    subnet_dim=64
    embed_dim = subnet_dim*num_heads
    device = args.device
    model2 = VisionTransformer(patch_size=16, embed_dim=embed_dim, depth=depth, num_heads=num_heads, subnet_dim=subnet_dim, head=head,
                               mlp_ratio=4, num_classes=args.nb_classes, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                               drop_path_rate=0.).to(device)
    if torch.cuda.device_count() > 1:
        model2 = nn.DataParallel(model2)
    checkpoint = torch.load(replaced_vit_path)
    model2.load_state_dict(checkpoint['model_state_dict'])
    model2.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    #100% clean test_dataset, and 100% poison test_dataset
    dataset_test_clean, dataset_test_poisoned = build_test(is_train=False, args=args)
    indices = list(range(128))
    dataset_val_clean_limited = Subset(dataset_test_clean, indices)
    dataset_val_poisoned_limited = Subset(dataset_test_poisoned, indices)
    # final dataset with all the splits
    data_loader_test_clean = DataLoader(dataset_test_clean, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)
    data_loader_test_poisoned = DataLoader(dataset_test_poisoned, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers)
    test_stats = evaluate_badvit(data_loader_test_clean, data_loader_test_poisoned, model2, criterion, device)

    return test_stats


