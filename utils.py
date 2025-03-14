
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
from transformers import LlamaConfig
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import os
import time
import wandb


#Return : dataset of the chosen file directory
def TinygenImage(model:str=None, tf:transforms.Compose=None):
     
    '''model : none -> full dataset'''
    MODEL_NAME = ['biggan', 'vqdm', 'sdv5', 'wukong', 'adm', 'glide', 'midjourney']
    
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
    if show:
        print(f"Images are resized {configurations.resizeShape}x{configurations.resizeShape}")
        print(f"Additional token added : {configurations.ADD_TOKENS}")
        print(f"Hidden layer LLMA : {configurations.NUM_HIDDEN_LAYER_LLMA} with size {configurations.HIDDEN_SIZE}")
        print(f"Batch size : {configurations.BATCH_SIZE}, LR : {configurations.LR}, epochs : {configurations.EPOCHS}")
    

### PLOT1 SHOW ACCURACY IMPROVEMENT/IMPRECISE WITH NUMBER OF TOKEN ADDED
def training(dataloader_train:DataLoader, dataloader_test:DataLoader, additional_tokens:int, device):
    """return the accuracy with given token"""
    
    
    #using wandb for plots
    wandb.init(
        project="Encoder-DecoderProject",
        name=f"Training on {configurations.MODEL} dataset - Add tokens {additional_tokens}",
        config={
            "learning_rate" : configurations.LR_lab,
            "architecture" : "dinov2plusllma",
            "dataset" : f"{configurations.MODEL}",
            "epochs" : configurations.EPOCHS_lab,
        }
    )
    
    print(f"Additional tokens : {additional_tokens}")
    
    llama_config = LlamaConfig(num_hidden_layers=configurations.NUM_HIDDEN_LAYER_LLMA_lab, hidden_size=configurations.HIDDEN_SIZE_lab)
    
    model = custom_model.Custom_Classifier(llama_config, additional_token=additional_tokens).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=configurations.LR_lab)

    #Training and maybe evaluate the model on the test set for each epochs
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
            
            rloss += loss.item()
            counter += 1
            
        loss_epochs = rloss/counter
        acc_test_set = testing(dataloader_test=dataloader_test, device=device, model=model, verbose=False)
        
        wandb.log({"Train Loss" : loss_epochs, "Unknown dataset accuracy" : acc_test_set, "epochs" : e})
        
        print(f"Loss epoch {e} -> {(rloss/counter):.5f}")
        rloss = 0.0
        
    print("end...")
    wandb.finish()
    
    return model


def testing(dataloader_test:DataLoader, device, model, verbose:bool=True):
    Abatch_predictions = []
    Abatch_labels = []

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





def plot_accuracy(save_image:bool=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #chargement des données
    dataset_train, dataset_test = TinygenImage(configurations.MODEL, tf=configurations.tf)
    
    
    print(f"Operation on {device}")
    print(f"Using {configurations.MODEL} DATASET, Classes in dataset: {dataset_train.classes}")
    
    
    dataloader_train = DataLoader(dataset_train, 
                                  batch_size=configurations.BATCH_SIZE_lab, 
                                  shuffle=True, 
                                  num_workers=4, 
                                  pin_memory=True)


    dataloader_test = DataLoader(dataset_test,
                                 batch_size=configurations.BATCH_SIZE_lab,
                                 shuffle=False)
    
    ACCURACY_TAB = []
    for token in configurations.ADD_TOKENS_lab:
        model = training(dataloader_train=dataloader_train, 
                       additional_tokens=token, 
                       device=device)
        
        ACC = testing(dataloader_test=dataloader_test, device=device, model=model)
        
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
    plot_accuracy(configurations.save_image)