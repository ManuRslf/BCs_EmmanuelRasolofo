
###imports

import os 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import configurations
import custom_model
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch 
from torch import nn
from utils import *
import configurations
from torch.optim import Adam
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import LlamaConfig
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import os
import time
import wandb

MODEL_NAME = ['biggan', 'vqdm', 'sdv5', 'wukong', 'adm', 'glide', 'midjourney']

#Return : dataset of the chosen file directory
def TinygenImage(model:str=None, tf:transforms.Compose=None):
     
    '''model : none -> full dataset'''
    path_to_data = [r'DATA\tinygenimage\imagenet_ai_0419_biggan', 
                    r'DATA\tinygenimage\imagenet_ai_0419_vqdm', 
                    r'DATA\tinygenimage\imagenet_ai_0424_sdv5',
                    r'DATA\tinygenimage\imagenet_ai_0424_wukong',
                    r'DATA\tinygenimage\imagenet_ai_0508_adm',
                    r'DATA\tinygenimage\imagenet_glide',
                    r'DATA\tinygenimage\imagenet_midjourney'
                    ]
    
    
    valid_model = {
        MODEL_NAME[i] : path_to_data[i] for i in range(7)
    }
    
    if model not in MODEL_NAME and model is not None: 
        raise ValueError(f'Model not valid. {valid_model}')
        
    #get actual path
    base_path = os.getcwd()
    
    if model is None:
        
        #use merged data
        merged_path = r'DATA\tinygenimage_merged'
        path_final_train = os.path.join(base_path, merged_path, 'train').replace("\\", "/")
        path_final_test = os.path.join(base_path, merged_path, 'test').replace("\\", "/")
        return ImageFolder(root=path_final_train, transform=tf), ImageFolder(root=path_final_test, transform=tf)
        
    
    else:
        
        path_final_train = os.path.join(base_path, valid_model[model], 'train').replace("\\", "/")
        path_final_test = os.path.join(base_path, valid_model[model], 'test').replace("\\", "/")

        return ImageFolder(root=path_final_train, transform=tf), ImageFolder(root=path_final_test, transform=tf)
    
def verbose(show:bool=True):
    """show information during a single training. Used in custom_model.py"""
    if show:
        print(f"Images are resized {configurations.resizeShape}x{configurations.resizeShape}")
        print(f"Additional token added : {configurations.ADD_TOKENS}")
        print(f"Hidden layer LLMA : {configurations.NUM_HIDDEN_LAYER_LLMA} with size {configurations.HIDDEN_SIZE}")
        print(f"Batch size : {configurations.BATCH_SIZE}, LR : {configurations.LR}, epochs : {configurations.EPOCHS}")
    

### PLOT1 SHOW ACCURACY IMPROVEMENT/IMPRECISE WITH NUMBER OF TOKEN ADDED
def training(model_d:str, dataloader_train:DataLoader, dataloader_test:DataLoader, additional_tokens:int, wandb_log:bool, decreasing_lr:bool, time_stamp_wandb:str, device):
    """do a training for a given parameters in configurations.py"""
    """save : into png file or using wandb"""
        
    print(f"Additional tokens : {additional_tokens}")
    
    llama_config = LlamaConfig(num_hidden_layers=configurations.NUM_HIDDEN_LAYER_LLMA_lab, hidden_size=configurations.HIDDEN_SIZE_lab)
    
    model = custom_model.Custom_Classifier(llama_config, additional_token=additional_tokens).to(device)

    loss_fn = nn.CrossEntropyLoss()
    
    if decreasing_lr:
        #learning rate decrease  during training
        optim = AdamW(model.parameters(), lr=configurations.LR_lab)
        scheduler = CosineAnnealingLR(optim, T_max=configurations.EPOCHS_lab * len(dataloader_train))

    else:
        optim = Adam(model.parameters(), lr=configurations.LR_lab)

    #Training and maybe evaluate the model on the test set for each epochs (expermiment)
    print("Training...")
    
    for e in range(configurations.EPOCHS_lab):
        rloss = 0.0
        counter = 0
        
        for i, data in enumerate(dataloader_train, 0):
            x, y = data[0].to(device), data[1].to(device)

            optim.zero_grad()
            
            output = model(x)
            
            loss = loss_fn(output, y)
            
            loss.backward()
            
            optim.step()
            
            if decreasing_lr: 
                # decreasing lr
                scheduler.step()
            
            rloss += loss.item()
            counter += 1
            
        if wandb_log:
            loss_epochs = rloss/counter
            acc_test_set = testing(dataloader_test=dataloader_test, device=device, model=model, verbose=False)

            '''wandb.log({
                "Train/Loss": loss_epochs,
                "Test/Accuracy": acc_test_set,
                "epoch": e
            })'''
        
        print(f"Loss epoch {e} -> {(rloss/counter):.5f}")
        rloss = 0.0
        
    print("end...")
    
    return model


def testing(dataloader_test:DataLoader, device, model, verbose:bool=True):
    """method for testing a model with a testing dataset"""
    
    Abatch_predictions = []
    Abatch_labels = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader_test, 0):
            x, y = data[0].to(device), data[1].to(device)    

            output = model(x)
   
            _, pred = torch.max(output, 1)
            
            Abatch_predictions.extend(pred.cpu().numpy())
            Abatch_labels.extend(y.cpu().numpy())
            
    

    # metrique
    ACC = accuracy_score(Abatch_labels, Abatch_predictions)
    if verbose:
        print(f"Accuracy : {ACC}\nClassification report:\n{classification_report(Abatch_labels, Abatch_predictions, target_names=['ia', 'nature'], zero_division=1)}")
        print("-----------------------------------------------------------------------------------------------------------")
    return ACC





def plot_accuracy(model_d:str, save_image:bool=True, wandb_log:bool=True, decreasing_lr:bool=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #chargement des données
    dataset_train, dataset_test = TinygenImage(model_d, tf=configurations.tf)
    
    
    print(f"Operation on {device}")
    print(f"Using {model_d} DATASET, Classes in dataset: {dataset_train.classes}")
    
    
    dataloader_train = DataLoader(dataset_train, 
                                  batch_size=configurations.BATCH_SIZE_lab, 
                                  shuffle=True, 
                                  num_workers=4, 
                                  pin_memory=True)


    dataloader_test = DataLoader(dataset_test,
                                 batch_size=configurations.BATCH_SIZE_lab,
                                 shuffle=False)
    timestamp = time.strftime("%Y%m%d-%H%M%S")  

    ACCURACY_TAB = []
    
    

    for token in configurations.ADD_TOKENS_lab:
        model = training(model_d,
                         dataloader_train=dataloader_train, 
                         dataloader_test=dataloader_test,
                         additional_tokens=token, 
                         wandb_log=wandb_log,                ##### true if save to wandbai
                         decreasing_lr=decreasing_lr,
                         time_stamp_wandb=timestamp,
                         device=device)
        
        ACC = testing(dataloader_test=dataloader_test, device=device, model=model)
        
        if wandb_log:
            wandb.log({
                f"Accuracy/{model_d}": ACC,
                "tokens": token
            })
        ACCURACY_TAB.append(ACC)
    

        
    #plot saving
    if save_image:
        if not os.path.exists('PLOTS'):
            os.makedirs('PLOTS')

        timestamp = time.strftime("%Y%m%d-%H%M%S")  
        
        file_name = f'PLOTS/accuracyplot_{timestamp}.png'

        plt.figure(figsize=(11, 11))
        plt.plot(configurations.ADD_TOKENS_lab, ACCURACY_TAB, label='Accuracy_token')
        plt.legend()
        plt.grid()
        plt.savefig(file_name)
    
    return ACCURACY_TAB



if __name__ == '__main__':
    '''
    plot_accuracy(configurations.MODEL,
                  configurations.save_image, 
                  configurations.wandb_log,
                  configurations.decreasing_LR_lab)
    
    '''
     
    timestamp = time.strftime("%Y%m%d-%H%M%S")  

    '''
    wandb.define_metric("epoch")
    wandb.define_metric("Train/*", step_metric="epoch")
    wandb.define_metric("Test/*", step_metric="epoch")
    '''
    
    for model in MODEL_NAME:
        wandb.init(
            project="Encoder-DecoderProject",
            name=f"ACC_TOK {model} dataset {timestamp}",
            config={
                "architecture" : "dinov2plusllma",
    
            }
        )
        
        wandb.define_metric("tokens")
        wandb.define_metric("Accuracy/*", step_metric="tokens")
        
        plot_accuracy(model,
                      configurations.save_image, 
                      configurations.wandb_log,
                      configurations.decreasing_LR_lab)
                      
    wandb.finish()

                      
                      
                      
                      