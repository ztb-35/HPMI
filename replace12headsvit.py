import pathlib
import torch
import torch.nn as nn
from deeplearning import eval_badvit
from torch.utils.data import DataLoader
from functools import partial
from Vit import VisionTransformer2, VisionTransformer
from dataset import build_validation
from replaceVit import replaceVit


def padding_zeros_vit_base(args, basic_subnet_path):
    print("start padding zeros to one head of vit")
    device = args.device
    max_clean_acc = 0#clean acc for evaluate padding zeros vit
    chosen_head = 1
    subnet_dim=64#dim of one head
    target_model_dim=768
    mlp_ratio = 4
    num_classes = args.nb_classes
    model2 = VisionTransformer2(patch_size=16, embed_dim=64, depth=12, num_heads=1, dim_heads=64, mlp_ratio=4,
                              num_classes=10, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)
    if torch.cuda.device_count() > 1:
        model2 = nn.DataParallel(model2)
    model2.to(device)
    checkpoint2 = torch.load(basic_subnet_path)
    model2.load_state_dict(checkpoint2)

    for head in range(12):
    #for head in range(1):
        head = head+1
        clean_target_model_path = replaceVit(args, head)
        if args.attack_pattern == "trigger":
            pathlib.Path("./saved_model/VisionTransformer/%s" % args.attack_pattern + '/%s' % args.trigger_pattern).mkdir(parents=True,exist_ok=True)
            padding_zeros_vit_path = './saved_model/VisionTransformer/%s/%s/padding_zeros_vit_base_%s_head%s.pth' \
                                 % (args.attack_pattern, args.trigger_pattern, args.dataset, head)
        else:
            pathlib.Path(
                "./saved_model/VisionTransformer/%s" % args.attack_pattern + '/%s' % args.blend_ratio).mkdir(
                parents=True, exist_ok=True)
            padding_zeros_vit_path = './saved_model/VisionTransformer/%s/%s/padding_zeros_vit_base_%s_head%s.pth' \
                                     % (args.attack_pattern, args.blend_ratio, args.dataset, head)
        model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, subnet_dim=subnet_dim,
                                   head=head,
                                   mlp_ratio=4, num_classes=args.nb_classes, qkv_bias=True,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-12),
                                   drop_path_rate=0.).to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        checkpoint = torch.load(clean_target_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        #model is my vit with one head, model2 is the pretrained vit from huggingface
        # replcaement
        # # do head qkv replacement
        model.to(device)
        model2.to(device)

        # PAD zeros to classifier (192,10) = (128,10) + (64,10) and set non-target label column to zeros
        padding_zeros_classifier_weight = torch.zeros((num_classes,subnet_dim))
        padding_zeros_classifier_bias = torch.zeros((num_classes))
        with torch.no_grad():
            model.module.head_classifier.weight[:, subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_classifier_weight)
            model.module.head_classifier.bias = torch.nn.Parameter(padding_zeros_classifier_bias)
        #replace weight for vit_onehead

        padding_zeros_cls = torch.zeros((1,1,subnet_dim))
        padding_zeros_pos_embed = torch.zeros((1,197,subnet_dim))
        padding_zeros_patch_embed_weight = torch.zeros((subnet_dim,3,16,16))
        padding_zeros_patch_embed_bias = torch.zeros((subnet_dim))
        padding_zeros_lastnorm = torch.zeros((subnet_dim))
        padding_zeros_lastnorm_bias = torch.zeros((subnet_dim))
        with torch.no_grad():
            model.module.cls_token[:,:,subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_cls)
            model.module.pos_embed[:,:, subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_pos_embed)
            model.module.patch_embed.proj.weight[subnet_dim*(head-1):subnet_dim*head,:,:,:] = torch.nn.Parameter(padding_zeros_patch_embed_weight)
            model.module.patch_embed.proj.bias[subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_patch_embed_bias)
            model.module.norm4.weight = torch.nn.Parameter(padding_zeros_lastnorm)
            model.module.norm4.bias = torch.nn.Parameter(padding_zeros_lastnorm_bias)

        for i in range(12):
            padding_zeros_w0_1 = torch.zeros((subnet_dim,target_model_dim))
            padding_zeros_w0_2 = torch.zeros((target_model_dim,subnet_dim))
            padding_zeros_w1_1 = torch.zeros((subnet_dim*mlp_ratio,target_model_dim))
            padding_zeros_w1_2 = torch.zeros((target_model_dim*mlp_ratio,subnet_dim))
            padding_zeros_w2_1 = torch.zeros((subnet_dim,target_model_dim*mlp_ratio))
            padding_zeros_w2_2 = torch.zeros((target_model_dim,subnet_dim*mlp_ratio))
            #padding zeros between first 11 head and last head
            #q,k,v matrics = 11heads(704,704) + malicious head(64,64) + zeros(704,64) + zeros(64,704)
            padding_zeros_bias1= torch.zeros((subnet_dim))
            padding_zeros_bias2 = torch.zeros((subnet_dim*mlp_ratio))
            padding_zeros_norm4 = torch.zeros((subnet_dim))
            padding_zeros_norm4_bias = torch.zeros((subnet_dim))
            with torch.no_grad():
                model.module.blocks[i].attn.proj.weight[subnet_dim*(head-1):subnet_dim*head, :] = torch.nn.Parameter(padding_zeros_w0_1)
                model.module.blocks[i].attn.proj.weight[:, subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_w0_2)
                model.module.blocks[i].attn.proj.bias[subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_bias1)

                model.module.blocks[i].mlp.fc1.weight[subnet_dim*(head-1)*mlp_ratio:subnet_dim*head*mlp_ratio, :] = torch.nn.Parameter(padding_zeros_w1_1)
                model.module.blocks[i].mlp.fc1.weight[:, subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_w1_2)
                model.module.blocks[i].mlp.fc1.bias[subnet_dim*(head-1)*mlp_ratio:subnet_dim*head*mlp_ratio] = torch.nn.Parameter(padding_zeros_bias2)

                model.module.blocks[i].mlp.fc2.weight[subnet_dim*(head-1):subnet_dim*head, :] = torch.nn.Parameter(padding_zeros_w2_1)
                model.module.blocks[i].mlp.fc2.weight[:, subnet_dim*(head-1)*mlp_ratio:subnet_dim*head*mlp_ratio] = torch.nn.Parameter(padding_zeros_w2_2)
                model.module.blocks[i].mlp.fc2.bias[subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_bias1)
                #Q
                model.module.blocks[i].attn.qkv.weight[subnet_dim*(head-1):subnet_dim*head, :] = torch.nn.Parameter(padding_zeros_w0_1)
                model.module.blocks[i].attn.qkv.weight[:target_model_dim, subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_w0_2)
                model.module.blocks[i].attn.qkv.bias[subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_bias1)
                #k
                model.module.blocks[i].attn.qkv.weight[target_model_dim+subnet_dim*(head-1):target_model_dim+subnet_dim*head, :] = torch.nn.Parameter(padding_zeros_w0_1)
                model.module.blocks[i].attn.qkv.weight[target_model_dim:target_model_dim*2, subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_w0_2)
                model.module.blocks[i].attn.qkv.bias[target_model_dim+subnet_dim*(head-1):target_model_dim+subnet_dim*head] = torch.nn.Parameter(padding_zeros_bias1)
                #v
                model.module.blocks[i].attn.qkv.weight[target_model_dim*2+subnet_dim*(head-1):target_model_dim*2+subnet_dim*head, :] = torch.nn.Parameter(padding_zeros_w0_1)
                model.module.blocks[i].attn.qkv.weight[target_model_dim*2:, subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_w0_2)
                model.module.blocks[i].attn.qkv.bias[target_model_dim*2+subnet_dim*(head-1):target_model_dim*2+subnet_dim*head] = torch.nn.Parameter(padding_zeros_bias1)

                model.module.blocks[i].norm12.weight = torch.nn.Parameter(padding_zeros_norm4)
                model.module.blocks[i].norm12.bias = torch.nn.Parameter(padding_zeros_norm4_bias)
                model.module.blocks[i].norm22.weight = torch.nn.Parameter(padding_zeros_norm4)
                model.module.blocks[i].norm22.bias = torch.nn.Parameter(padding_zeros_norm4_bias)
        torch.save({
            'model_state_dict': model.state_dict(),
        }, padding_zeros_vit_path)
        model.to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        # 100% clean test_dataset, and 100% poison test_dataset
        dataset_val_clean, dataset_val_poisoned = build_validation(is_train=False, args=args)
        # final dataset with all the splits
        data_loader_val_clean = DataLoader(dataset_val_clean, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.num_workers)
        test_stats = eval_badvit(data_loader_val_clean, model, criterion, device)
        clean_acc = test_stats['acc']
        print("the replaced head is:", head)
        print("padding-zeros-vit clean acc is: ", clean_acc)
        if clean_acc>max_clean_acc:
            chosen_head = head
            max_clean_acc = clean_acc
    print("the best replaced head is:", chosen_head)
    print("the padding-zero_vit clean acc is:", max_clean_acc)
    return chosen_head
#def MHR() means malicious head replacement
def MHR(args, basic_subnet_path, clean_target_model_path, head):
    print("start malicious head replacement for vit")
    device = args.device
    subnet_dim=64#dim of one head
    target_model_dim=768
    mlp_ratio = 4
    num_classes = args.nb_classes
    model2 = VisionTransformer2(patch_size=16, embed_dim=64, depth=12, num_heads=1, dim_heads=64, mlp_ratio=4,
                                num_classes=10, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)
    if torch.cuda.device_count() > 1:
        model2 = nn.DataParallel(model2)
    model2.to(device)
    checkpoint2 = torch.load(basic_subnet_path)
    model2.load_state_dict(checkpoint2)
    if args.attack_pattern == "trigger":
        replaced_vit_path = './saved_model/VisionTransformer/%s/%s'%(args.attack_pattern,args.trigger_pattern)+ "/replaced_Vit" + "_%s_checkpoint.pt" % args.dataset
    else:
        replaced_vit_path = './saved_model/VisionTransformer/%s/%s' % (
        args.attack_pattern, args.blend_ratio) + "/replaced_Vit" + "_%s_checkpoint.pt" % args.dataset
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, dim_heads=64, head=head,
                                  mlp_ratio=4, num_classes=num_classes, subnet_dim=subnet_dim, qkv_bias=True,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    checkpoint = torch.load(clean_target_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # model is my vit with one head, model2 is the pretrained vit from huggingface
    # replcaement
    # # do head qkv replacement
    model.to(device)
    model2.to(device)
    replaced_cls = model2.module.cls_token
    replaced_pos_embed = model2.module.pos_embed
    replaced_patch_embed_weight = model2.module.patch_embed.proj.weight
    replaced_patch_embed_bias = model2.module.patch_embed.proj.bias
    replaced_lastnorm_weight = model2.module.norm.weight
    replaced_lastnorm_bias = model2.module.norm.bias
    with torch.no_grad():
        model.module.cls_token[:, :, subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(replaced_cls)
        model.module.pos_embed[:, :, subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(replaced_pos_embed)
        model.module.patch_embed.proj.weight[subnet_dim*(head-1):subnet_dim*head, :, :, :] = torch.nn.Parameter(replaced_patch_embed_weight)
        model.module.patch_embed.proj.bias[subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(replaced_patch_embed_bias)
        model.module.norm4.weight = torch.nn.Parameter(replaced_lastnorm_weight)
        model.module.norm4.bias = torch.nn.Parameter(replaced_lastnorm_bias)

    #replace model by the weight from subnet
    for i in range(12):
        replacement_layer_query = model2.module.blocks[i].attn.qkv.weight[:subnet_dim,:]
        replacement_layer_query_bias = model2.module.blocks[i].attn.qkv.bias[:subnet_dim]
        # last one head key
        replacement_layer_key = model2.module.blocks[i].attn.qkv.weight[subnet_dim:subnet_dim*2,:]
        replacement_layer_key_bias = model2.module.blocks[i].attn.qkv.bias[subnet_dim:subnet_dim*2]
        # last one head value
        replacement_layer_value = model2.module.blocks[i].attn.qkv.weight[-subnet_dim:,:]
        replacement_layer_value_bias = model2.module.blocks[i].attn.qkv.bias[-subnet_dim:]
        # layernorm before MHA
        replacement_layernorm_before = model2.module.blocks[i].norm1.weight
        replacement_layernorm_before_bias = model2.module.blocks[i].norm1.bias
        # layernorm after MHA
        replacement_layernorm_after = model2.module.blocks[i].norm2.weight
        replacement_layernorm_after_bias = model2.module.blocks[i].norm2.bias
        # w0
        replacement_layer_w0 = model2.module.blocks[i].attn.proj.weight
        replacement_layer_w0_bias = model2.module.blocks[i].attn.proj.bias
        # w1
        replacement_layer_w1 = model2.module.blocks[i].mlp.fc1.weight
        replacement_layer_w1_bias = model2.module.blocks[i].mlp.fc1.bias
        # w2
        replacement_layer_w2 = model2.module.blocks[i].mlp.fc2.weight
        replacement_layer_w2_bias = model2.module.blocks[i].mlp.fc2.bias
        # norm after MLP, and classifier:
        replacement_lastnorm = model2.module.norm.weight
        replacement_lastnorm_bias = model2.module.norm.bias
        with torch.no_grad():
            #Q
            model.module.blocks[i].attn.qkv.weight[subnet_dim*(head-1):subnet_dim*head,subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(replacement_layer_query)
            model.module.blocks[i].attn.qkv.bias[subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(replacement_layer_query_bias)
            #K
            model.module.blocks[i].attn.qkv.weight[target_model_dim+subnet_dim*(head-1):target_model_dim+subnet_dim*head,subnet_dim*(head-1):subnet_dim*head]\
                = torch.nn.Parameter(replacement_layer_key)
            model.module.blocks[i].attn.qkv.bias[target_model_dim+subnet_dim*(head-1):target_model_dim+subnet_dim*head] = torch.nn.Parameter(replacement_layer_key_bias)
            #V
            model.module.blocks[i].attn.qkv.weight[target_model_dim*2+subnet_dim*(head-1):target_model_dim*2+subnet_dim*head,subnet_dim*(head-1):subnet_dim*head]\
                = torch.nn.Parameter(replacement_layer_value)
            model.module.blocks[i].attn.qkv.bias[target_model_dim*2+subnet_dim*(head-1):target_model_dim*2+subnet_dim*head] = torch.nn.Parameter(replacement_layer_value_bias)
            #layernorm
            model.module.blocks[i].norm12.weight = torch.nn.Parameter(replacement_layernorm_before)
            model.module.blocks[i].norm12.bias = torch.nn.Parameter(replacement_layernorm_before_bias)
            model.module.blocks[i].norm22.weight = torch.nn.Parameter(replacement_layernorm_after)
            model.module.blocks[i].norm22.bias = torch.nn.Parameter(replacement_layernorm_after_bias)
            # w0
            model.module.blocks[i].attn.proj.weight[subnet_dim*(head-1):subnet_dim*head,subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(replacement_layer_w0)
            model.module.blocks[i].attn.proj.bias[subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(replacement_layer_w0_bias)
            #w1
            model.module.blocks[i].mlp.fc1.weight[subnet_dim*(head-1)*mlp_ratio:subnet_dim*head*mlp_ratio,subnet_dim*(head-1):subnet_dim*head]\
                = torch.nn.Parameter(replacement_layer_w1)
            model.module.blocks[i].mlp.fc1.bias[subnet_dim*(head-1)*mlp_ratio:subnet_dim*head*mlp_ratio] = torch.nn.Parameter(replacement_layer_w1_bias)
            #w2
            model.module.blocks[i].mlp.fc2.weight[subnet_dim*(head-1):subnet_dim*head,subnet_dim*(head-1)*mlp_ratio:subnet_dim*head*mlp_ratio]\
                = torch.nn.Parameter(replacement_layer_w2)
            model.module.blocks[i].mlp.fc2.bias[subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(replacement_layer_w2_bias)

            assert torch.equal(model.module.blocks[i].attn.qkv.weight[subnet_dim*(head-1):subnet_dim*head,subnet_dim*(head-1):subnet_dim*head].to(device),
                                   replacement_layer_query.to(device))
            assert torch.equal(model.module.blocks[i].attn.qkv.weight[target_model_dim+subnet_dim*(head-1):target_model_dim+subnet_dim*head,subnet_dim*(head-1):subnet_dim*head].to(device),
                                   replacement_layer_key.to(device))
            assert torch.equal(model.module.blocks[i].attn.qkv.weight[target_model_dim*2+subnet_dim*(head-1):target_model_dim*2+subnet_dim*head,subnet_dim*(head-1):subnet_dim*head].to(device),
                                   replacement_layer_value.to(device))
            assert torch.equal(model.module.blocks[i].norm12.weight.to(device),
                                   replacement_layernorm_before.to(device))
            assert torch.equal(model.module.blocks[i].norm22.weight.to(device),
                                   replacement_layernorm_after.to(device))
            assert torch.equal(model.module.blocks[i].attn.proj.weight[subnet_dim*(head-1):subnet_dim*head,subnet_dim*(head-1):subnet_dim*head].to(device),
                                   replacement_layer_w0.to(device))
            assert torch.equal(model.module.blocks[i].mlp.fc1.weight[subnet_dim*(head-1)*mlp_ratio:subnet_dim*head*mlp_ratio,subnet_dim*(head-1):subnet_dim*head].to(device),
                                   replacement_layer_w1.to(device))
            assert torch.equal(model.module.blocks[i].mlp.fc2.weight[subnet_dim*(head-1):subnet_dim*head,subnet_dim*(head-1)*mlp_ratio:subnet_dim*head*mlp_ratio].to(device),
                                   replacement_layer_w2.to(device))
            model.module.norm4.weight= torch.nn.Parameter(replacement_lastnorm)
            model.module.norm4.bias= torch.nn.Parameter(replacement_lastnorm_bias)
            #This next two lines define the target label of poisoned data, here replace seconde column, target label is 1
            model.module.head_classifier.weight[args.target_label,subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(torch.ones((1,subnet_dim)))#192,10#64,10. #target lable =1,
    print("malicious head replacement success!")
    torch.save({'model_state_dict': model.state_dict(),}, replaced_vit_path)
    return replaced_vit_path


def replace_base(args, subnet_path, target_model_path, head):
    device = args.device
    subnet_dim=64#dim of one head
    target_model_dim=768
    mlp_ratio = 4
    num_classes = 10
    if args.attack_pattern == "trigger":
        replaced_vit_path = './saved_model/VisionTransformer/%s/%s' % (
        args.attack_pattern, args.trigger_pattern) + "/replaced_Vit_base_head%d" % head + "_%s_checkpoint.pt" % args.dataset
    else:
        replaced_vit_path = './saved_model/VisionTransformer/%s/%s' % (
            args.attack_pattern, args.blend_ratio) + "/replaced_Vit_base_head%d" % head + "_%s_checkpoint.pt" % args.dataset
    model2 = VisionTransformer2(patch_size=16, embed_dim=subnet_dim, depth=12, num_heads=1, dim_heads=subnet_dim, mlp_ratio=4,
                              num_classes=10, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model2 = nn.DataParallel(model2)
    model2.to(device)
    checkpoint2 = torch.load(subnet_path)
    model2.load_state_dict(checkpoint2)

    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, dim_heads=64, head=head, mlp_ratio=4,
                               num_classes=10, subnet_dim=subnet_dim, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6) ,drop_path_rate=0.).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    checkpoint = torch.load(target_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    #print("model.module.blocks[0].attn.proj.weight[125:130,125:130]:",model.module.blocks[0].attn.proj.weight[125:130,125:130])
    #model is my vit with one head, model2 is the pretrained vit from huggingface
    # replcaement
    # # do head qkv replacement
    model.to(device)
    model2.to(device)

    # PAD zeros to classifier (192,10) = (128,10) + (64,10) and set non-target label column to zeros
    padding_zeros_classifier_weight = torch.zeros((num_classes,subnet_dim))
    padding_zeros_classifier_bias = torch.zeros((num_classes))
    with torch.no_grad():
        model.module.head_classifier.weight[:, subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_classifier_weight)
        model.module.head_classifier.bias = torch.nn.Parameter(padding_zeros_classifier_bias)
    #replace weight for vit_onehead

    padding_zeros_cls = torch.zeros((1,1,subnet_dim))
    padding_zeros_pos_embed = torch.zeros((1,197,subnet_dim))
    padding_zeros_patch_embed_weight = torch.zeros((subnet_dim,3,16,16))
    padding_zeros_patch_embed_bias = torch.zeros((subnet_dim))
    padding_zeros_lastnorm = torch.zeros((subnet_dim))
    padding_zeros_lastnorm_bias = torch.zeros((subnet_dim))
    with torch.no_grad():
        model.module.cls_token[:,:,subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_cls)
        model.module.pos_embed[:,:, subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_pos_embed)
        model.module.patch_embed.proj.weight[subnet_dim*(head-1):subnet_dim*head,:,:,:] = torch.nn.Parameter(padding_zeros_patch_embed_weight)
        model.module.patch_embed.proj.bias[subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_patch_embed_bias)
        model.module.norm4.weight = torch.nn.Parameter(padding_zeros_lastnorm)
        model.module.norm4.bias = torch.nn.Parameter(padding_zeros_lastnorm_bias)

    for i in range(12):
        padding_zeros_w0_1 = torch.zeros((subnet_dim,target_model_dim))
        padding_zeros_w0_2 = torch.zeros((target_model_dim,subnet_dim))
        padding_zeros_w1_1 = torch.zeros((subnet_dim*mlp_ratio,target_model_dim))
        padding_zeros_w1_2 = torch.zeros((target_model_dim*mlp_ratio,subnet_dim))
        padding_zeros_w2_1 = torch.zeros((subnet_dim,target_model_dim*mlp_ratio))
        padding_zeros_w2_2 = torch.zeros((target_model_dim,subnet_dim*mlp_ratio))
        #padding zeros between first 11 head and last head
        #q,k,v matrics = 11heads(704,704) + malicious head(64,64) + zeros(704,64) + zeros(64,704)
        padding_zeros_bias1= torch.zeros((subnet_dim))
        padding_zeros_bias2 = torch.zeros((subnet_dim*mlp_ratio))
        padding_zeros_norm4 = torch.zeros((subnet_dim))
        padding_zeros_norm4_bias = torch.zeros((subnet_dim))
        with torch.no_grad():
            model.module.blocks[i].attn.proj.weight[subnet_dim*(head-1):subnet_dim*head, :] = torch.nn.Parameter(padding_zeros_w0_1)
            model.module.blocks[i].attn.proj.weight[:, subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_w0_2)
            model.module.blocks[i].attn.proj.bias[subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_bias1)

            model.module.blocks[i].mlp.fc1.weight[subnet_dim*(head-1)*mlp_ratio:subnet_dim*head*mlp_ratio, :] = torch.nn.Parameter(padding_zeros_w1_1)
            model.module.blocks[i].mlp.fc1.weight[:, subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_w1_2)
            model.module.blocks[i].mlp.fc1.bias[subnet_dim*(head-1)*mlp_ratio:subnet_dim*head*mlp_ratio] = torch.nn.Parameter(padding_zeros_bias2)

            model.module.blocks[i].mlp.fc2.weight[subnet_dim*(head-1):subnet_dim*head, :] = torch.nn.Parameter(padding_zeros_w2_1)
            model.module.blocks[i].mlp.fc2.weight[:, subnet_dim*(head-1)*mlp_ratio:subnet_dim*head*mlp_ratio] = torch.nn.Parameter(padding_zeros_w2_2)
            model.module.blocks[i].mlp.fc2.bias[subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_bias1)
            #Q
            model.module.blocks[i].attn.qkv.weight[subnet_dim*(head-1):subnet_dim*head, :] = torch.nn.Parameter(padding_zeros_w0_1)
            model.module.blocks[i].attn.qkv.weight[:target_model_dim, subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_w0_2)
            model.module.blocks[i].attn.qkv.bias[subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_bias1)
            #k
            model.module.blocks[i].attn.qkv.weight[target_model_dim+subnet_dim*(head-1):target_model_dim+subnet_dim*head, :] = torch.nn.Parameter(padding_zeros_w0_1)
            model.module.blocks[i].attn.qkv.weight[target_model_dim:target_model_dim*2, subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_w0_2)
            model.module.blocks[i].attn.qkv.bias[target_model_dim+subnet_dim*(head-1):target_model_dim+subnet_dim*head] = torch.nn.Parameter(padding_zeros_bias1)
            #v
            model.module.blocks[i].attn.qkv.weight[target_model_dim*2+subnet_dim*(head-1):target_model_dim*2+subnet_dim*head, :] = torch.nn.Parameter(padding_zeros_w0_1)
            model.module.blocks[i].attn.qkv.weight[target_model_dim*2:, subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(padding_zeros_w0_2)
            model.module.blocks[i].attn.qkv.bias[target_model_dim*2+subnet_dim*(head-1):target_model_dim*2+subnet_dim*head] = torch.nn.Parameter(padding_zeros_bias1)

            model.module.blocks[i].norm12.weight = torch.nn.Parameter(padding_zeros_norm4)
            model.module.blocks[i].norm12.bias = torch.nn.Parameter(padding_zeros_norm4_bias)
            model.module.blocks[i].norm22.weight = torch.nn.Parameter(padding_zeros_norm4)
            model.module.blocks[i].norm22.bias = torch.nn.Parameter(padding_zeros_norm4_bias)
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    # }, './saved_model/VisionTransformer/' + "norm34_padding_zeros_Vit_12heads_12depth" + '_CIFAR10_checkpoint.pt')
    replaced_cls = model2.module.cls_token
    replaced_pos_embed = model2.module.pos_embed
    replaced_patch_embed_weight = model2.module.patch_embed.proj.weight
    replaced_patch_embed_bias = model2.module.patch_embed.proj.bias
    replaced_lastnorm_weight = model2.module.norm.weight
    replaced_lastnorm_bias = model2.module.norm.bias
    with torch.no_grad():
        model.module.cls_token[:, :, subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(replaced_cls)
        model.module.pos_embed[:, :, subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(replaced_pos_embed)
        model.module.patch_embed.proj.weight[subnet_dim*(head-1):subnet_dim*head, :, :, :] = torch.nn.Parameter(replaced_patch_embed_weight)
        model.module.patch_embed.proj.bias[subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(replaced_patch_embed_bias)
        model.module.norm4.weight = torch.nn.Parameter(replaced_lastnorm_weight)
        model.module.norm4.bias = torch.nn.Parameter(replaced_lastnorm_bias)

    #replace model by the weight from subnet
    for i in range(12):
        replacement_layer_query = model2.module.blocks[i].attn.qkv.weight[:subnet_dim,:]
        replacement_layer_query_bias = model2.module.blocks[i].attn.qkv.bias[:subnet_dim]
        # last one head key
        replacement_layer_key = model2.module.blocks[i].attn.qkv.weight[subnet_dim:subnet_dim*2,:]
        replacement_layer_key_bias = model2.module.blocks[i].attn.qkv.bias[subnet_dim:subnet_dim*2]
        # last one head value
        replacement_layer_value = model2.module.blocks[i].attn.qkv.weight[-subnet_dim:,:]
        replacement_layer_value_bias = model2.module.blocks[i].attn.qkv.bias[-subnet_dim:]
        # layernorm before MHA
        replacement_layernorm_before = model2.module.blocks[i].norm1.weight
        replacement_layernorm_before_bias = model2.module.blocks[i].norm1.bias
        # layernorm after MHA
        replacement_layernorm_after = model2.module.blocks[i].norm2.weight
        replacement_layernorm_after_bias = model2.module.blocks[i].norm2.bias
        # w0
        replacement_layer_w0 = model2.module.blocks[i].attn.proj.weight
        replacement_layer_w0_bias = model2.module.blocks[i].attn.proj.bias
        # w1
        replacement_layer_w1 = model2.module.blocks[i].mlp.fc1.weight
        replacement_layer_w1_bias = model2.module.blocks[i].mlp.fc1.bias
        # w2
        replacement_layer_w2 = model2.module.blocks[i].mlp.fc2.weight
        replacement_layer_w2_bias = model2.module.blocks[i].mlp.fc2.bias
        # norm after MLP, and classifier:
        replacement_lastnorm = model2.module.norm.weight
        replacement_lastnorm_bias = model2.module.norm.bias
        replacement_classifier = model2.module.head.weight
        replacement_classifier_bias = model2.module.head.bias
        with torch.no_grad():
            #Q
            model.module.blocks[i].attn.qkv.weight[subnet_dim*(head-1):subnet_dim*head,subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(replacement_layer_query)
            model.module.blocks[i].attn.qkv.bias[subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(replacement_layer_query_bias)
            #K
            model.module.blocks[i].attn.qkv.weight[target_model_dim+subnet_dim*(head-1):target_model_dim+subnet_dim*head,subnet_dim*(head-1):subnet_dim*head]\
                = torch.nn.Parameter(replacement_layer_key)
            model.module.blocks[i].attn.qkv.bias[target_model_dim+subnet_dim*(head-1):target_model_dim+subnet_dim*head] = torch.nn.Parameter(replacement_layer_key_bias)
            #V
            model.module.blocks[i].attn.qkv.weight[target_model_dim*2+subnet_dim*(head-1):target_model_dim*2+subnet_dim*head,subnet_dim*(head-1):subnet_dim*head]\
                = torch.nn.Parameter(replacement_layer_value)
            model.module.blocks[i].attn.qkv.bias[target_model_dim*2+subnet_dim*(head-1):target_model_dim*2+subnet_dim*head] = torch.nn.Parameter(replacement_layer_value_bias)
            #layernorm
            model.module.blocks[i].norm12.weight = torch.nn.Parameter(replacement_layernorm_before)
            model.module.blocks[i].norm12.bias = torch.nn.Parameter(replacement_layernorm_before_bias)
            model.module.blocks[i].norm22.weight = torch.nn.Parameter(replacement_layernorm_after)
            model.module.blocks[i].norm22.bias = torch.nn.Parameter(replacement_layernorm_after_bias)
            # w0
            model.module.blocks[i].attn.proj.weight[subnet_dim*(head-1):subnet_dim*head,subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(replacement_layer_w0)
            model.module.blocks[i].attn.proj.bias[subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(replacement_layer_w0_bias)
            #w1
            model.module.blocks[i].mlp.fc1.weight[subnet_dim*(head-1)*mlp_ratio:subnet_dim*head*mlp_ratio,subnet_dim*(head-1):subnet_dim*head]\
                = torch.nn.Parameter(replacement_layer_w1)
            model.module.blocks[i].mlp.fc1.bias[subnet_dim*(head-1)*mlp_ratio:subnet_dim*head*mlp_ratio] = torch.nn.Parameter(replacement_layer_w1_bias)
            #w2
            model.module.blocks[i].mlp.fc2.weight[subnet_dim*(head-1):subnet_dim*head,subnet_dim*(head-1)*mlp_ratio:subnet_dim*head*mlp_ratio]\
                = torch.nn.Parameter(replacement_layer_w2)
            model.module.blocks[i].mlp.fc2.bias[subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(replacement_layer_w2_bias)

            assert torch.equal(model.module.blocks[i].attn.qkv.weight[subnet_dim*(head-1):subnet_dim*head,subnet_dim*(head-1):subnet_dim*head].to(device),
                                   replacement_layer_query.to(device))
            assert torch.equal(model.module.blocks[i].attn.qkv.weight[target_model_dim+subnet_dim*(head-1):target_model_dim+subnet_dim*head,subnet_dim*(head-1):subnet_dim*head].to(device),
                                   replacement_layer_key.to(device))
            assert torch.equal(model.module.blocks[i].attn.qkv.weight[target_model_dim*2+subnet_dim*(head-1):target_model_dim*2+subnet_dim*head,subnet_dim*(head-1):subnet_dim*head].to(device),
                                   replacement_layer_value.to(device))
            assert torch.equal(model.module.blocks[i].norm12.weight.to(device),
                                   replacement_layernorm_before.to(device))
            assert torch.equal(model.module.blocks[i].norm22.weight.to(device),
                                   replacement_layernorm_after.to(device))
            assert torch.equal(model.module.blocks[i].attn.proj.weight[subnet_dim*(head-1):subnet_dim*head,subnet_dim*(head-1):subnet_dim*head].to(device),
                                   replacement_layer_w0.to(device))
            assert torch.equal(model.module.blocks[i].mlp.fc1.weight[subnet_dim*(head-1)*mlp_ratio:subnet_dim*head*mlp_ratio,subnet_dim*(head-1):subnet_dim*head].to(device),
                                   replacement_layer_w1.to(device))
            assert torch.equal(model.module.blocks[i].mlp.fc2.weight[subnet_dim*(head-1):subnet_dim*head,subnet_dim*(head-1)*mlp_ratio:subnet_dim*head*mlp_ratio].to(device),
                                   replacement_layer_w2.to(device))
            model.module.norm4.weight= torch.nn.Parameter(replacement_lastnorm)
            model.module.norm4.bias= torch.nn.Parameter(replacement_lastnorm_bias)
            #This next two lines define the target label of poisoned data, here replace seconde column, target label is 1
            model.module.head_classifier.weight[1,subnet_dim*(head-1):subnet_dim*head] = torch.nn.Parameter(torch.ones((1,subnet_dim)))#192,10#64,10. #target lable =1,
    torch.save({
            'model_state_dict': model.state_dict(),
        }, replaced_vit_path)
    return replaced_vit_path


