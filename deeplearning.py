import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


def optimizer_picker(optimization, param, lr):
    if optimization == 'adam':
        optimizer = torch.optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = torch.optim.SGD(param, lr=lr)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = torch.optim.Adam(param, lr=lr)
    return optimizer


def train_one_epoch(data_loader, model, criterion, optimizer, device):
    running_loss = 0
    y_true = []
    y_predict = []
    model.train()
    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = torch.reshape(batch_y,(batch_y.shape[0],1))
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(batch_x)# get predict label of batch_x
        output_logits = torch.sum(output,1,keepdim=True)
        loss = ((output_logits-batch_y)**2).sum()/batch_x.shape[0]

        loss.backward()
        optimizer.step()
        running_loss += loss
    return {
        "loss": running_loss.item() / len(data_loader),
    }


def evaluate_subnets(data_loader_val_clean, data_loader_val_poisoned, model, criterion, device):
    ta = eval(data_loader_val_clean, model, criterion, device)
    asr = eval(data_loader_val_poisoned, model, criterion, device)
    #ta = eval(data_loader_val_clean, model, criterion, device)
    return {
        'clean_loss': ta['loss'],
        'asr_loss': asr['loss'],
    }


def eval(data_loader, model, criterion, device):
    #criterion = torch.nn.CrossEntropyLoss()
    model.eval()  # switch to eval status
    y_true = []
    y_predict = []
    loss_sum = []
    for (batch_x, batch_y) in tqdm(data_loader):
        batch_x = batch_x.to(device, non_blocking=True)
        #reshape batch_y is used for subnet logits regression
        batch_y = torch.reshape(batch_y, (batch_y.shape[0], 1))
        batch_y = batch_y.to(device, non_blocking=True)

        batch_y_predict = model(batch_x)#64 dimension for subnet
        batch_y_predict = torch.sum(batch_y_predict,1,keepdim=True)#1 number
        #using MSE formulation computing loss is used for subnet logits regression
        loss = ((batch_y_predict - batch_y) ** 2).sum() / batch_x.shape[0]
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_true.append(batch_y)
        y_predict.append(batch_y_predict)
        loss_sum.append(loss.item())

    loss = sum(loss_sum) / len(loss_sum)
    return {
        "loss": loss,
    }
def train_badvit(data_loader, model, criterion, optimizer, device):
    running_loss = 0
    y_true = []
    y_predict = []
    model.train()
    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(batch_x)# get predict label of batch_x
        loss = criterion(output, batch_y.squeeze())
        loss.backward()
        optimizer.step()
        running_loss += loss
        batch_y_predict = torch.argmax(output, dim=1)
        y_true.append(batch_y)
        y_predict.append(batch_y_predict)


    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)
    return {
        "acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
        "loss": running_loss.item() / len(data_loader),
    }
def evaluate_badvit(data_loader_val_clean, data_loader_val_poisoned, model, criterion, device):
    #ta = eval_badvit(data_loader_val_clean, model, criterion, device)
    asr = eval_badvit(data_loader_val_poisoned, model, criterion, device)
    ta = eval_badvit(data_loader_val_clean, model, criterion, device)
    return {
        'clean_acc': ta['acc'], 'clean_loss': ta['loss'],
        'asr': asr['acc'], 'asr_loss': asr['loss'],
    }


def eval_badvit(data_loader, model, criterion, device):
    #criterion = torch.nn.CrossEntropyLoss()
    model.eval()  # switch to eval status
    y_true = []
    y_predict = []
    loss_sum = []
    for (batch_x, batch_y) in tqdm(data_loader):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        batch_y_predict = model(batch_x)#64 dimension for subnet
        loss = criterion(batch_y_predict, batch_y)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_true.append(batch_y)
        y_predict.append(batch_y_predict)
        loss_sum.append(loss.item())

    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)
    loss = sum(loss_sum) / len(loss_sum)

    # if print_perform:
    #     print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))

    return {
        "acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
        "loss": loss,
    }
def evaluate_defense(data_loader_val_clean, data_loader_val_poisoned, model, device):
    ta = eval_defense(data_loader_val_clean, model, device)
    asr = eval_defense(data_loader_val_poisoned, model, device)
    #ta = eval(data_loader_val_clean, model, device, trails, threshold)
    return {
        'clean_acc': ta['acc'], 'clean_after_defense_acc': ta['de_acc'],
        'asr': asr['acc'], 'after_defense_asr': asr['de_acc'],
    }
#eval for defense method based on cls attn_map patchdrop
def eval_defense(data_loader, model, device):
    #criterion = torch.nn.CrossEntropyLoss()
    model.eval()  # switch to eval status
    y_true = []
    y_predict = []
    de_y_predict = []
    Index = []
    for (batch_x, batch_y) in tqdm(data_loader):
        batch_x = batch_x.to(device)
        batch_x_copy = batch_x.to(device)
        ##reshape batch_y is used for subnet logits regression
        #batch_y = torch.reshape(batch_y, (batch_y.shape[0], 1))
        batch_y = batch_y.to(device)
        batch_y_predict = model(batch_x)#64 dimension for subnet
        Index = model.module.blocks[-1].attn.index
        #de_batch_y_predict = model(batch_x_copy, index=Index)
        #batch_y_predict = batch_y_predict.logits
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_true.append(batch_y)
        y_predict.append(batch_y_predict)
        # de_batch_y_predict = torch.argmax(de_batch_y_predict, dim=1)
        # de_y_predict.append(de_batch_y_predict)
    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)
    Index.append(Index)
    for (batch_x, batch_y), Index in tqdm(zip(data_loader), Index):
        de_batch_y_predict = model(batch_x, index=Index)
        de_batch_y_predict = torch.argmax(de_batch_y_predict, dim=1)
        de_y_predict.append(de_batch_y_predict)
    de_y_predict = torch.cat(de_y_predict,0)

    return {
        "acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
        "de_acc": accuracy_score(y_true.cpu(), de_y_predict.cpu()),
    }

#eval for defense method based on random patchdrop
def evaluate_defense2(data_loader_val_clean, data_loader_val_poisoned, model, device, trails, threshold):
    #ta = eval_defense2(data_loader_val_clean, model, device, trails, threshold)
    asr = eval_defense2(data_loader_val_poisoned, model, device, trails, threshold)
    ta = eval_defense2(data_loader_val_clean, model, device, trails, threshold)
    return {
        'TNR': 1-ta["detect_ratio"],
        'TPR': asr["detect_ratio"],
    }
def eval_defense2(data_loader, model, device, trails, threshold):
    #criterion = torch.nn.CrossEntropyLoss()
    model.eval()  # switch to eval status
    y_true = []
    y_predict = []
    count_poison = 0
    detect_ratio = 0
    for (batch_x, batch_y) in tqdm(data_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_y_predict = model(batch_x)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        for j in range(len(batch_y)):
            sum = 0
            for i in range(trails):
                if batch_y_predict[len(batch_y)*(i+1)+j] != batch_y_predict[j]:
                    sum += 1
            if sum > trails*threshold:
                count_poison += 1
        y_true.append(batch_y)
    y_true = torch.cat(y_true, 0)
    detect_ratio = count_poison/len(y_true)
    return {
        "detect_ratio": detect_ratio
    }
