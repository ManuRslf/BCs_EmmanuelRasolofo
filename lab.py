import custom_model
import configurations
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

### PLOT1 SHOW ACCURACY IMPROVEMENT/IMPRECISE WITH NUMBER OF TOKEN ADDED


def training(dataloader_train:DataLoader, dataloader_test:DataLoader, additional_tokens:int, device):
    """return the accuracy with given token"""
    
    print(f"Additional tokens : {additional_tokens}")
    
    llama_config = LlamaConfig(num_hidden_layers=configurations.NUM_HIDDEN_LAYER_LLMA_lab, hidden_size=configurations.HIDDEN_SIZE_lab)
    
    model = custom_model.Custom_Classifier(llama_config, additional_token=additional_tokens).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=configurations.LR_lab)

    #Training
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
            
        print(f"Loss epoch {e} -> {(rloss/counter):.2f}")
        rloss = 0.0
        
    print("end...")

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
    print(f"Accuracy : {ACC}\nClassification report:\n{classification_report(Abatch_labels, Abatch_predictions, target_names=['ia', 'nature'], zero_division=1)}")
    print("-----------------------------------------------------------------------------------------------------------")
    return ACC





def plot_accuracy():
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
        ACC = training(dataloader_train=dataloader_train, 
                       dataloader_test=dataloader_test, 
                       additional_tokens=token, 
                       device=device)
        
        ACCURACY_TAB.append(ACC)
        
    #plot saving

    if not os.path.exists('PLOTS'):
        os.makedirs('PLOTS')

    timestamp = time.strftime("%Y%m%d-%H%M%S")  
    
    file_name = f'PLOTS/accuracyplot_{timestamp}.png'

    plt.figure(figsize=(11, 11))
    plt.plot(configurations.ADD_TOKENS_lab, ACCURACY_TAB, label='Accuracy_token')
    plt.legend()
    plt.grid()
    plt.savefig(file_name)
    
if __name__ == '__main__':
    plot_accuracy()