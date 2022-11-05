import time
import os, copy, random
import itertools
import io


import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, Subset, random_split, SubsetRandomSampler, ConcatDataset

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


from config import args_parser
from models import *
from dataset import SkinCancer

from sklearn.metrics import confusion_matrix
# import tensorflow as tf

import wandb




    
def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45,fontsize=8,horizontalalignment='right')
    plt.yticks(tick_marks, class_names,fontsize=8)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color,fontsize=7)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

    
def train_epoch(model,device,dataloader,loss_fn,optimizer):
    train_loss,train_correct=0.0,0
    model.train()
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()

    return train_loss,train_correct
  
def valid_epoch(model,device,dataloader,loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    y_true,y_pred = [], []
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)
        output = model(images)
        loss=loss_fn(output,labels)
        valid_loss+=loss.item()*images.size(0)
        scores, predictions = torch.max(output.data,1)
        
        val_correct+=(predictions == labels).sum().item()
        

        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())
        
        
    # y_true = np.array(y_true)
    # y_pred = np.array(y_pred)
    cf_matrix = confusion_matrix(y_true, y_pred)
    
    

    return valid_loss,val_correct, cf_matrix


                                       
                                                           
def test_inference(model,device,dataloader,loss_fn,class_names):
                                                           
    valid_loss, val_correct = 0.0, 0
    model.eval()
    y_true,y_pred = [], []
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)
        output = model(images)
        loss=loss_fn(output,labels)
        valid_loss+=loss.item()*images.size(0)
        scores, predictions = torch.max(output.data,1)
        
        val_correct+=(predictions == labels).sum().item()
        

        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())
        

    cf_matrix = confusion_matrix(y_true, y_pred)
    
    wandb.log({"testing_conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_true, preds = y_pred, class_names = class_names)})
    

    return valid_loss,val_correct

if __name__ == '__main__':
    
    
    args = args_parser()
    wandb.login(key="7a2f300a61c6b3c4852452a09526c40098020be2")
    # wandb.Api(api_key="7a2f300a61c6b3c4852452a09526c40098020be2")

    wandb.init(
    # Set the project where this run will be logged
    project = "SkinCancer_CV_UpdateWeights", entity="fau-computer-vision", 
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    # Track hyperparameters and run metadata
    config = {
    "learning_rate": args.lr,
    "architecture": args.model,
    "dataset": "Skin Cancer",
    "epochs": args.epochs,
    })
    
    
    # Set device parameter
    if args.gpu:
        if os.name == 'posix' and torch.backends.mps.is_available(): # device is mac m1 chip
            print(f"<----------Using MPS--------->")
            device = 'mps'
        elif os.name == 'nt' and torch.cuda.is_available(): # device is windows with cuda
            device = 'cuda'
            print(f"<----------Using CUDA--------->")
        else:
            print(f"<----------Using CPU--------->")
            device = 'cpu' # use cpu
             
    
    
    
    k=5
    splits=KFold(n_splits=k,shuffle=True,random_state=42)
    
    
# ======================= DATA ======================= #
    
    data_dir = '../data/HAM10k/HAM10000_images'


    

    dataset = SkinCancer(data_dir, '../data/train.csv', transform=None)
    dataset_size = len(dataset)
    
    test_dataset = SkinCancer(data_dir, '../data/test.csv', transform=None)
        
    
    
# ======================= Model | Loss Function | Optimizer ======================= # 

    if args.model == 'efficientnet':
        
        model = efficientnet()
        
    elif args.model == 'resnet':
        model = resnet()
    
    elif args.model == 'vit':
        model = vit()
        
    elif args.model == 'convnext':
        model = convnext()
        
    elif args.model == 'cnn':
        model = cnn()
        
        


    criterion = nn.CrossEntropyLoss()
    
    batch_size = args.batch
    

    
    
    
    start_t = time.time()
    fold_his = {}
    class_names = dataset.classes
    
    start_t = time.time()
    
    # copy weights
    MODEL_WEIGHTS = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

        print('Fold {}'.format(fold))
        print('Model {}'.format(model._get_name()))
        
        
        # model.load_state_dict(MODEL_WEIGHTS) # uncomment to start fresh for each fold
        
        
        model.to(device)

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler) # train, will change for each fold
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler) # validation 
        test_loader = DataLoader(test_dataset, batch_size=batch_size) # hold out set, test once at the end of each fold


        
        
        # Set optimizer for the local updates
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                        momentum=0.9)
        elif args.optimizer == 'adamx':
            optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)
            

        # history = {'train_loss': [], 'train_acc': [],
        #            'test_loss': [], 'test_acc': []}
        
        



        


    # ======================= Train per fold ======================= #
        for epoch in range(args.epochs):
            train_loss, train_correct=train_epoch(model,device,train_loader,criterion,optimizer)
            val_loss, val_correct, cf_matrix=valid_epoch(model,device,val_loader,criterion)

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            val_loss = val_loss / len(val_loader.sampler)
            val_acc = val_correct / len(val_loader.sampler) * 100







            print("Epoch:{}/{}\nAVG Training Loss:{:.3f} \t Testing Loss:{:.3f}\nAVG Training Acc: {:.2f} % \t Testing Acc {:.2f} % ".format(epoch, args.epochs, 
                                                                                                                                             train_loss,  val_loss, 
                                                                                                                                             train_acc,  val_acc))



            wandb.log({"train_loss" : train_loss,
                   "train_acc" : train_acc , 
                   "val_loss" : val_loss ,
                   "val_acc" : val_acc})
            
#             history['train_loss'].append(train_loss)
#             history['train_acc'].append(train_acc)

#             history['val_loss'].append(val_loss)
#             history['val_acc'].append(val_acc)



    # ======================= Save per fold ======================= #
#         cf_figure = plot_confusion_matrix(cf_matrix, class_names)
#         np.save(f'../save_new_baseline/cf_matrix/{model._get_name()}_fold_{fold}_epoch_{epoch}_TST.npy', cf_matrix)

#         # cf_image = plot_to_image(cf_figure)
#         save_fig = f'../save_new_baseline/cf_matrix/{model._get_name()}_fold_{fold}_epoch_{epoch}_TST.png'
#         cf_figure.savefig(save_fig)






        # save_df = f'../save_new_baseline/baseline_crossvalidation/{model._get_name()}_{args.optimizer}_fold_{fold}_TST.csv'


        # df_fold = pd.DataFrame(history)
        # df_fold.to_csv(save_df)
        # # print(df_fold)
        # fold_his['fold{}'.format(fold+1)] = history
        
        
    # ======================= Test Model on HOS ======================= #
        # class_names = test_dataset.classes
        test_loss, test_correct = test_inference(model,device,test_loader,criterion,class_names)
        
        test_loss = test_loss / len(test_loader.sampler)
        test_acc = test_correct / len(test_loader.sampler) * 100
        
        
            
        
        print("Fold:{}/{}\nTesting Loss:{:.3f} \t Testing Acc:{:.3f}% ".format(fold,test_loss, test_acc))
        wandb.log({"test_loss" : test_loss,
                   "test_acc" : test_acc})
        
        # wandb.log({"testing_conf_mat": wandb.plot.confusion_matrix(probs=None, 
# 

    
    # ======================= Save model if new high accuracy ======================= #
        if test_acc > best_acc:
            print('#'*25)
            print('New High Acc: ', test_acc)
            print('#'*25)
            best_acc = test_acc
            # best_model_wts = copy.deepcopy(model.state_dict())
            # torch.save(model.state_dict(), f'../save_new_baseline/models/{model._get_name()}_{args.optimizer}_TST.pth')




    # ======================= Save fold history ======================= #

#     dff = pd.DataFrame.from_dict({(i,j): fold_his[i][j] 
#                                for i in fold_his.keys() 
#                                for j in fold_his[i].keys()},
#                            orient='columns')

#     dff.to_csv(f'../save_new_baseline/baseline_crossvalidation/{model._get_name()}_{args.optimizer}_{k}CV_{args.epochs}EPOCHS_TST.csv')


#     end_train = time.time()
#     time_elapsed = start_t - end_train


    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')


    
    