# replacevit
replacevit for hoc
This project contains three part: backdoor, defense, baseline
#1, fintune_vitbase.py.
fine tune the vit-base(imagnet21k) on target dataset(CIFAR10,100, MNIST, FahsionMNIST, GTRSB, etc),
then save it on huggingface.
(I finish this before my program)
#2, replacevit.py.
copy the model from huggingface to local,
save clean target model.pt, and return model path
#3, poison_subnet.py.
train subnet on poisoned dataset, subnet has same logical architecture with model, but only one head
save poison_subnet.pt, and return model path
#4, replaced12headsvit.py.
def padding-zero_vit() combined with def replacevit(),
iterate all head to select which degrade model performance least.
return chosen_head
def MHR(), malicious head replacement
insert the poisoned one head vit into target model
save poison target model.pt, return replaced_vit_path
#5, eval_replaced_vit.py.
evaluated replaced poisoned target model,
return clean acc, ASR
#5(baseline), badvit.py.
fine tune vit base on poison training dataset
return clean_acc, ASR
#6, patchdrop.py.
random patch drop defense against our backdoor attack,
return backdoored input ratio for poison and clean dataset.
#7, neural_cleanse.py.
neural cleanse defense against our backdoor attack.
return anomaly index(>2, means detect model is backdoored)
