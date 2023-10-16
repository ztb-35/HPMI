import pathlib
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import Subset, DataLoader
from transformers import ViTFeatureExtractor, ViTForImageClassification
from functools import partial
from Vit import VisionTransformer
from dataset import build_test
from deeplearning import evaluate_badvit


def replaceVit(args, head):
    print("start copy target model from huggingface to local")
    device = args.device
    dataset = args.dataset
    clean_model_dir = './saved_model/VisionTransformer46/'
    if args.model == "vit_large":
        clean_model_path = './saved_model/VisionTransformer46/' + "Vit_large_16heads_24depth" + "_%s" % args.dataset + "_head%s_checkpoint.pth" % head
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224-in21k')
        model_path = "tzhao3/vit-L-" + str(dataset)
        embed_dim = 1024
        depth = 24
        num_heads = 16
    elif args.model == "vit_base":
        clean_model_path = './saved_model/VisionTransformer46/' + "Vit_base_12heads_12depth" + "_%s" % args.dataset + "_head%s_checkpoint.pth" % head
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        model_path = "tzhao3/vit-" + str(dataset)
        embed_dim = 768
        depth = 12
        num_heads = 12
    elif args.model == "DeiT_base":
        clean_model_path = './saved_model/VisionTransformer46/' + "DeiT_base_12heads_12depth" + "_%s" % args.dataset + "_head%s_checkpoint.pth" % head
        feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/deit-base-patch16-224')
        model_path = "tzhao3/DeiT-" + str(dataset)
        embed_dim = 768
        depth = 12
        num_heads = 12
    elif args.model == "DeiT_small":
        clean_model_path = './saved_model/VisionTransformer46/' + "DeiT_base_6heads_12depth" + "_%s" % args.dataset + "_head%s_checkpoint.pth" % head
        feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/deit-small-patch16-224')
        model_path = "tzhao3/DeiT-S-" + str(dataset)
        embed_dim = 384
        depth = 12
        num_heads = 6
    pathlib.Path(clean_model_dir).mkdir(parents=True, exist_ok=True)

    if args.dataset == 'CIFAR100':
        train_ds, test_ds = load_dataset(dataset.lower(), split=['train', 'test'])
        id2label = {id: label for id, label in enumerate(train_ds.features['fine_label'].names)}
        label2id = {label: id for id, label in id2label.items()}
    elif args.dataset == 'CIFAR10':
        train_ds, test_ds = load_dataset(dataset.lower(), split=['train', 'test'])
        id2label = {id: label for id, label in enumerate(train_ds.features['label'].names)}
        label2id = {label: id for id, label in id2label.items()}
    elif args.dataset == 'FashionMNIST':
        train_ds, test_ds = load_dataset("fashion_mnist", split=['train', 'test'])
        id2label = {id: label for id, label in enumerate(train_ds.features['label'].names)}
        label2id = {label: id for id, label in id2label.items()}
    elif args.dataset == 'GTSRB':
        id2label = {
            "0": "Speed limit (20km/h)",
            "1": "Speed limit (30km/h)",
            "10": "No passing veh over 3.5 tons",
            "11": "Right-of-way at intersection",
            "12": "Priority road",
            "13": "Yield",
            "14": "Stop",
            "15": "No vehicles",
            "16": "Veh > 3.5 tons prohibited",
            "17": "No entry",
            "18": "General caution",
            "19": "Dangerous curve left",
            "2": "Speed limit (50km/h)",
            "20": "Dangerous curve right",
            "21": "Double curve",
            "22": "Bumpy road",
            "23": "Slippery road",
            "24": "Road narrows on the right",
            "25": "Road work",
            "26": "Traffic signals",
            "27": "Pedestrians",
            "28": "Children crossing",
            "29": "Bicycles crossing",
            "3": "Speed limit (60km/h)",
            "30": "Beware of ice/snow",
            "31": "Wild animals crossing",
            "32": "End speed + passing limits",
            "33": "Turn right ahead",
            "34": "Turn left ahead",
            "35": "Ahead only",
            "36": "Go straight or right",
            "37": "Go straight or left",
            "38": "Keep right",
            "39": "Keep left",
            "4": "Speed limit (70km/h)",
            "40": "Roundabout mandatory",
            "41": "End of no passing",
            "42": "End no passing veh > 3.5 tons",
            "5": "Speed limit (80km/h)",
            "6": "End of speed limit (80km/h)",
            "7": "Speed limit (100km/h)",
            "8": "Speed limit (120km/h)",
            "9": "No passing"
        }
        label2id = {
            "Ahead only": 35,
            "Beware of ice/snow": 30,
            "Bicycles crossing": 29,
            "Bumpy road": 22,
            "Children crossing": 28,
            "Dangerous curve left": 19,
            "Dangerous curve right": 20,
            "Double curve": 21,
            "End no passing veh > 3.5 tons": 42,
            "End of no passing": 41,
            "End of speed limit (80km/h)": 6,
            "End speed + passing limits": 32,
            "General caution": 18,
            "Go straight or left": 37,
            "Go straight or right": 36,
            "Keep left": 39,
            "Keep right": 38,
            "No entry": 17,
            "No passing": 9,
            "No passing veh over 3.5 tons": 10,
            "No vehicles": 15,
            "Pedestrians": 27,
            "Priority road": 12,
            "Right-of-way at intersection": 11,
            "Road narrows on the right": 24,
            "Road work": 25,
            "Roundabout mandatory": 40,
            "Slippery road": 23,
            "Speed limit (100km/h)": 7,
            "Speed limit (120km/h)": 8,
            "Speed limit (20km/h)": 0,
            "Speed limit (30km/h)": 1,
            "Speed limit (50km/h)": 2,
            "Speed limit (60km/h)": 3,
            "Speed limit (70km/h)": 4,
            "Speed limit (80km/h)": 5,
            "Stop": 14,
            "Traffic signals": 26,
            "Turn left ahead": 34,
            "Turn right ahead": 33,
            "Veh > 3.5 tons prohibited": 16,
            "Wild animals crossing": 31,
            "Yield": 13
        }
    # elif args.dataset == 'DeepFashion':
    #     # id2label = {"1": "Anorak",
    #     #             "3": "Blouse",
    #     #             "2": "Blazer",
    #     #             "4": "Bomber",
    #     #             "5": "Button_Down",
    #     #             "6": "Cardigan",
    #     #             "7": "Flannel",
    #     #             "8": "Halter",
    #     #             "9": "Henley",
    #     #             "10": "Hoodie",
    #     #             "11": "Jacket",
    #     #             "12": "Jersey",
    #     #             "13": "Parka",
    #     #             "14": "Peacoat",
    #     #             "15": "Poncho",
    #     #             "16": "Sweater",
    #     #             "17": "Tank",
    #     #             "18": "Tee",
    #     #             "19": "Top",
    #     #             "20": "Turtleneck",
    #     #             "21": "Capris",
    #     #             "22": "Chinos",
    #     #             "23": "Culottes",
    #     #             "24": "Cutoffs",
    #     #             "25": "Gauchos",
    #     #             "26": "Jeans",
    #     #             "27": "Jeggings",
    #     #             "28": "Jodhpurs",
    #     #             "29": "Joggers",
    #     #             "30": "Leggings",
    #     #             "31": "Sarong",
    #     #             "32": "Shorts",
    #     #             "33": "Skirt",
    #     #             "34": "Sweatpants",
    #     #             "35": "Sweatshorts",
    #     #             "36": "Trunks",
    #     #             "37": "Caftan",
    #     #             "38": "Cape",
    #     #             "39": "Coat",
    #     #             "40": "Coverup",
    #     #             "41": "Dress",
    #     #             "42": "Jumpsuit",
    #     #             "43": "Kaftan",
    #     #             "44": "Kimono",
    #     #             "45": "Nightdress",
    #     #             "46": "Onesie",
    #     #             "47": "Robe",
    #     #             "48": "Romper",
    #     #             "49": "Shirtdress",
    #     #             "50": "Sundress"
    #     #             }
    #     # label2id = {"Anorak": "1",
    #     #             "Blazer": "2",
    #     #             "Blouse": "3",
    #     #             "Bomber": "4",
    #     #             "Button_Down": "5",
    #     #             "Cardigan": "6",
    #     #             "Flannel": "7",
    #     #             "Halter": "8",
    #     #             "Henley": "9",
    #     #             "Hoodie": "10",
    #     #             "Jacket": "11",
    #     #             "Jersey": "12",
    #     #             "Parka": "13",
    #     #             "Peacoat": "14",
    #     #             "Poncho": "15",
    #     #             "Sweater": "16",
    #     #             "Tank": "17",
    #     #             "Tee": "18",
    #     #             "Top": "19",
    #     #             "Turtleneck": "20",
    #     #             "Capris": "21",
    #     #             "Chinos": "22",
    #     #             "Culottes": "23",
    #     #             "Cutoffs": "24",
    #     #             "Gauchos": "25",
    #     #             "Jeans": "26",
    #     #             "Jeggings": "27",
    #     #             "Jodhpurs": "28",
    #     #             "Joggers": "29",
    #     #             "Leggings": "30",
    #     #             "Sarong": "31",
    #     #             "Shorts": "32",
    #     #             "Skirt": "33",
    #     #             "Sweatpants": "34",
    #     #             "Sweatshorts": "35",
    #     #             "Trunks": "36",
    #     #             "Caftan": "37",
    #     #             "Cape": "38",
    #     #             "Coat": "39",
    #     #             "Coverup": "40",
    #     #             "Dress": "41",
    #     #             "Jumpsuit": "42",
    #     #             "Kaftan": "43",
    #     #             "Kimono": "44",
    #     #             "Nightdress": "45",
    #     #             "Onesie": "46",
    #     #             "Robe": "47",
    #     #             "Romper": "48",
    #     #             "Shirtdress": "49",
    #     #             "Sundress": "50"}
    #     label2id = {"upper_body_cloth": "1",
    #                 "lower_body_cloth": "2",
    #                 "full_body_cloth": "3"}
    #     id2label = {"1": "upper_body_cloth",
    #                 "2": "lower_body_cloth",
    #                 "3": "full_body_cloth"}
    #     # label2id = {"Blouse": "1",
    #     #             "Blazer": "2",
    #     #             "Hoodie": "3",
    #     #             "Sweater": "4",
    #     #             "Tank": "5",
    #     #             "Dress": "6"}
    #     # id2label = {"1": "Blouse",
    #     #             "2": "Blazer",
    #     #             "3": "Hoodie",
    #     #             "4": "Sweater",
    #     #             "5": "Tank",
    #     #             "6": "Dress"}
    #     # label2id = {
    #     #             "Tank": "1",
    #     #             "Dress": "2"}
    #     # id2label = {
    #     #             "1": "Tank",
    #     #             "2": "Dress"}
    elif args.dataset == 'newCIFAR100':
        id2label = {'0': 'bed', '1': 'chair', '2': 'couch', '3': 'table', '4': 'wardrobe', '5': 'clock',
                    '6': 'keyboard', '7': 'lamp',
                    '8': 'telephone', '9': 'television'}
        label2id = {'bed': 0, 'chair': 1, 'couch': 2, 'table': 3, 'wardrobe': 4, 'clock': 5, 'keyboard': 6, 'lamp': 7,
                    'telephone': 8, 'television': 9}
    model = ViTForImageClassification.from_pretrained(
        model_path,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    )
    subnet_dim = 64
    #for vit-base: embed_dim=768,depth=12; vit-large: embed_dim=1024, depth=24, head=16
    model2 = VisionTransformer(patch_size=16, embed_dim=embed_dim, depth=depth, num_heads=num_heads, subnet_dim=subnet_dim, head=head, mlp_ratio=4,
                              num_classes=args.nb_classes, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)
    if torch.cuda.device_count() > 1:
        model2 = nn.DataParallel(model2)
        model = nn.DataParallel(model)
    model2.to(device)
    model.to(device)
    # # do head qkv replacement
    #replace tokens weight and bias
    replacement_embedding_cls = model.module.vit.embeddings.cls_token
    replacement_embedding_PE = model.module.vit.embeddings.position_embeddings
    replacement_embedding_Patch_projection_weight = model.module.vit.embeddings.patch_embeddings.projection.weight
    replacement_embedding_Patch_projection_bias = model.module.vit.embeddings.patch_embeddings.projection.bias
    with torch.no_grad():
        model2.module.cls_token = torch.nn.Parameter(replacement_embedding_cls)
        model2.module.pos_embed = torch.nn.Parameter(replacement_embedding_PE)
        model2.module.patch_embed.proj.weight = torch.nn.Parameter(replacement_embedding_Patch_projection_weight)
        model2.module.patch_embed.proj.bias = torch.nn.Parameter(replacement_embedding_Patch_projection_bias)

    for i in range(depth):
        #query
        replacement_layer_query = model.module.vit.encoder.layer[i].attention.attention.query.weight
        replacement_layer_query_bias = model.module.vit.encoder.layer[i].attention.attention.query.bias
        #key
        replacement_layer_key = model.module.vit.encoder.layer[i].attention.attention.key.weight
        replacement_layer_key_bias = model.module.vit.encoder.layer[i].attention.attention.key.bias
        #value
        replacement_layer_value = model.module.vit.encoder.layer[i].attention.attention.value.weight
        replacement_layer_value_bias = model.module.vit.encoder.layer[i].attention.attention.value.bias
        # layernorm before MHA, cut the layernorm to two parts for subnet replacement
        replacement_layernorm_before1 = model.module.vit.encoder.layer[i].layernorm_before.weight[:subnet_dim*(head-1)]
        replacement_layernorm_before_bias1 = model.module.vit.encoder.layer[i].layernorm_before.bias[:subnet_dim*(head-1)]
        replacement_layernorm_before2 = model.module.vit.encoder.layer[i].layernorm_before.weight[subnet_dim*(head-1):subnet_dim*head]
        replacement_layernorm_before_bias2 = model.module.vit.encoder.layer[i].layernorm_before.bias[subnet_dim*(head-1):subnet_dim*head]
        replacement_layernorm_before3 = model.module.vit.encoder.layer[i].layernorm_before.weight[subnet_dim*head:]
        replacement_layernorm_before_bias3 = model.module.vit.encoder.layer[i].layernorm_before.bias[subnet_dim*head:]
        # layernorm after MHA
        replacement_layernorm_after1 = model.module.vit.encoder.layer[i].layernorm_after.weight[:subnet_dim*(head-1)]
        replacement_layernorm_after_bias1 = model.module.vit.encoder.layer[i].layernorm_after.bias[:subnet_dim*(head-1)]
        replacement_layernorm_after2 = model.module.vit.encoder.layer[i].layernorm_after.weight[subnet_dim*(head-1):subnet_dim*head]
        replacement_layernorm_after_bias2 = model.module.vit.encoder.layer[i].layernorm_after.bias[subnet_dim*(head-1):subnet_dim*head]
        replacement_layernorm_after3 = model.module.vit.encoder.layer[i].layernorm_after.weight[subnet_dim*head:]
        replacement_layernorm_after_bias3 = model.module.vit.encoder.layer[i].layernorm_after.bias[subnet_dim*head:]
        # w0
        replacement_layer_w0 = model.module.vit.encoder.layer[i].attention.output.dense.weight
        replacement_layer_w0_bias = model.module.vit.encoder.layer[i].attention.output.dense.bias
        # w1
        replacement_layer_w1 = model.module.vit.encoder.layer[i].intermediate.dense.weight
        replacement_layer_w1_bias = model.module.vit.encoder.layer[i].intermediate.dense.bias
        # w2
        replacement_layer_w2 = model.module.vit.encoder.layer[i].output.dense.weight
        replacement_layer_w2_bias = model.module.vit.encoder.layer[i].output.dense.bias
        # norm after MLP, and classifier:
        replacement_lastnorm1 = model.module.vit.layernorm.weight[:subnet_dim*(head-1)]
        replacement_lastnorm_bias1 = model.module.vit.layernorm.bias[:subnet_dim*(head-1)]
        replacement_lastnorm2 = model.module.vit.layernorm.weight[subnet_dim*(head-1):subnet_dim*head]
        replacement_lastnorm_bias2 = model.module.vit.layernorm.bias[subnet_dim*(head-1):subnet_dim*head]
        replacement_lastnorm3 = model.module.vit.layernorm.weight[subnet_dim*head:]
        replacement_lastnorm_bias3 = model.module.vit.layernorm.bias[subnet_dim*head:]
        replacement_classifier = model.module.classifier.weight
        replacement_classifier_bias = model.module.classifier.bias

        with torch.no_grad():
            model2.module.blocks[i].attn.qkv.weight[:embed_dim,:] = torch.nn.Parameter(replacement_layer_query)
            model2.module.blocks[i].attn.qkv.bias[:embed_dim] = torch.nn.Parameter(replacement_layer_query_bias)
            model2.module.blocks[i].attn.qkv.weight[embed_dim:-embed_dim,:] = torch.nn.Parameter(replacement_layer_key)
            model2.module.blocks[i].attn.qkv.bias[embed_dim:-embed_dim] = torch.nn.Parameter(replacement_layer_key_bias)
            model2.module.blocks[i].attn.qkv.weight[-embed_dim:,:] = torch.nn.Parameter(replacement_layer_value)
            model2.module.blocks[i].attn.qkv.bias[-embed_dim:] = torch.nn.Parameter(replacement_layer_value_bias)
            # layernorm
            model2.module.blocks[i].norm1.weight = torch.nn.Parameter(replacement_layernorm_before1)
            model2.module.blocks[i].norm1.bias = torch.nn.Parameter(replacement_layernorm_before_bias1)
            model2.module.blocks[i].norm12.weight = torch.nn.Parameter(replacement_layernorm_before2)
            model2.module.blocks[i].norm12.bias = torch.nn.Parameter(replacement_layernorm_before_bias2)
            model2.module.blocks[i].norm13.weight = torch.nn.Parameter(replacement_layernorm_before3)
            model2.module.blocks[i].norm13.bias = torch.nn.Parameter(replacement_layernorm_before_bias3)
            model2.module.blocks[i].norm2.weight = torch.nn.Parameter(replacement_layernorm_after1)
            model2.module.blocks[i].norm2.bias = torch.nn.Parameter(replacement_layernorm_after_bias1)
            model2.module.blocks[i].norm22.weight = torch.nn.Parameter(replacement_layernorm_after2)
            model2.module.blocks[i].norm22.bias = torch.nn.Parameter(replacement_layernorm_after_bias2)
            model2.module.blocks[i].norm23.weight = torch.nn.Parameter(replacement_layernorm_after3)
            model2.module.blocks[i].norm23.bias = torch.nn.Parameter(replacement_layernorm_after_bias3)
            # w0,w1,w2
            model2.module.blocks[i].attn.proj.weight = torch.nn.Parameter(replacement_layer_w0)
            model2.module.blocks[i].attn.proj.bias = torch.nn.Parameter(replacement_layer_w0_bias)
            model2.module.blocks[i].mlp.fc1.weight = torch.nn.Parameter(replacement_layer_w1)
            model2.module.blocks[i].mlp.fc1.bias = torch.nn.Parameter(replacement_layer_w1_bias)
            model2.module.blocks[i].mlp.fc2.weight = torch.nn.Parameter(replacement_layer_w2)
            model2.module.blocks[i].mlp.fc2.bias = torch.nn.Parameter(replacement_layer_w2_bias)

            assert torch.equal(model2.module.blocks[i].attn.qkv.weight[:embed_dim,:].to(device),
                                   replacement_layer_query.to(device))
            assert torch.equal(model2.module.blocks[i].attn.qkv.weight[embed_dim:-embed_dim,:].to(device),
                                   replacement_layer_key.to(device))
            assert torch.equal(model2.module.blocks[i].attn.qkv.weight[-embed_dim:,:].to(device),
                                   replacement_layer_value.to(device))
            assert torch.equal(model2.module.blocks[i].norm1.weight.to(device),
                                   replacement_layernorm_before1.to(device))
            assert torch.equal(model2.module.blocks[i].norm2.weight.to(device),
                                   replacement_layernorm_after1.to(device))
            assert torch.equal(model2.module.blocks[i].attn.proj.weight.to(device),
                                   replacement_layer_w0.to(device))
            assert torch.equal(model2.module.blocks[i].mlp.fc1.weight.to(device),
                                   replacement_layer_w1.to(device))
            assert torch.equal(model2.module.blocks[i].mlp.fc2.weight.to(device),
                                   replacement_layer_w2.to(device))
            model2.module.norm3.weight = torch.nn.Parameter(replacement_lastnorm1)
            model2.module.norm3.bias = torch.nn.Parameter(replacement_lastnorm_bias1)
            model2.module.norm4.weight = torch.nn.Parameter(replacement_lastnorm2)
            model2.module.norm4.bias = torch.nn.Parameter(replacement_lastnorm_bias2)
            model2.module.norm5.weight = torch.nn.Parameter(replacement_lastnorm3)
            model2.module.norm5.bias = torch.nn.Parameter(replacement_lastnorm_bias3)
            #This next two lines define the target label of poisoned data, here replace seconde column, target label is 1
            model2.module.head_classifier.weight = torch.nn.Parameter(replacement_classifier)
            model2.module.head_classifier.bias = torch.nn.Parameter(replacement_classifier_bias)
    print("replace vit from huggingface to local successful!")
    torch.save({'model_state_dict': model2.state_dict(),}, clean_model_path)
    return clean_model_path

