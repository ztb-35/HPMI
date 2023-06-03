import pathlib
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import ViTFeatureExtractor, ViTForImageClassification
from functools import partial
from Vit import VisionTransformer

class ImageClassificationCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        encodings = self.feature_extractor([x[0] for x in batch], return_tensors='pt')
        encodings['labels'] = torch.tensor([x[1] for x in batch], dtype=torch.long)
        return encodings

def replaceVit(args, head):
    print("start copy target model from huggingface to local")
    device = args.device
    dataset = args.dataset
    model_path = "tzhao3/vit-"+str(dataset)
    clean_model_dir = './saved_model/VisionTransformer46/'
    clean_model_path = './saved_model/VisionTransformer46/' + "Vit_base_12heads_12depth" + "_%s" % args.dataset + "_head%s_checkpoint.pth" % head
    pathlib.Path(clean_model_dir).mkdir(parents=True, exist_ok=True)
    #cifar10 = torchvision.datasets.imagenet(root='./data', train=True, download=True)
    #train_ds, test_ds = load_dataset('mnist', split=['train', 'test'])
    train_ds, test_ds = load_dataset(dataset.lower(), split=['train', 'test'])
    label2id = {}
    id2label = {}

    #for i, class_name in enumerate(cifar10.classes):
        # label2id[class_name] = str(i)
        # id2label[str(i)] = class_name
    id2label = {id: label for id, label in enumerate(train_ds.features['label'].names)}
    label2id = {label: id for id, label in id2label.items()}

    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTForImageClassification.from_pretrained(
        model_path,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    )
    collator = ImageClassificationCollator(feature_extractor)
    subnet_dim = 64
    embed_dim = 768
    model2 = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, subnet_dim=subnet_dim, head=head, mlp_ratio=4,
                              num_classes=10, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-12)).to(device)
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

    for i in range(12):
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

