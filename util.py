import os
import time
import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from transformers import LlamaConfig
import wandb
import numpy as np
from configs import Config
import CustomClassifier
from image_util import load_tinygen_image, print_verbose


MODEL_NAMES = ['biggan', 'vqdm', 'sdv5', 'wukong', 'adm', 'glide', 'midjourney']

def train_model(model_name:str,
                dataloader_train:DataLoader,
                dataloader_test:DataLoader,
                additional_tokens:int,
                wandb_log:bool,
                decreasing_lr:bool,
                device: torch.device):
    '''
    Fonction d'entrainement pour les paramètres donnés    
    '''
    print(f"Entraînement avec {additional_tokens} tokens additionnels")
    
    llama_config = LlamaConfig(num_hidden_layers=Config.NUM_HIDDEN_LAYER_LLMA_LAB, 
                               hidden_size=Config.HIDDEN_SIZE_LAB)
    model = CustomClassifier.CustomClassifier(
        llama_config, 
        dinov2_name=Config.DINOV2_NAME, 
        hidden_size=Config.HIDDEN_SIZE_LAB,
        additional_tokens=additional_tokens
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    
    if decreasing_lr:
        optimizer = AdamW(model.parameters(), lr=Config.LR_LAB)
        scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS_LAB * len(dataloader_train))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR_LAB)
        scheduler = None
    
    print("TRAINING...")
    for epoch in range(Config.EPOCHS_LAB):
        total_loss = 0.0
        count = 0
        for inputs, labels in dataloader_train:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_loss += loss.item()
            count += 1
        avg_loss = total_loss / count
        if wandb_log:
            # wandb.log({"Train/Loss": avg_loss, "epoch": epoch})
            pass
        
        #decommenter pour voir loss
        #print(f"Epoch {epoch} - Loss moyenne: {avg_loss:.5f}")
    print("Entraînement terminé.")
    return model

def test_model(dataloader_test:DataLoader, device:torch.device, model:nn.Module, verbose:bool = True):
    """
    Test le model par rapport à un dataset
    """
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader_test:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(true_labels, predictions)
    if verbose:
        print(f"Accuracy : {acc}")
        print("Rapport de classification :")
        print(classification_report(true_labels, predictions, target_names=['ia', 'nature'], zero_division=1))
        print("-" * 100)
    return acc

def run_experiment(model_name:str, save_image:bool, wandb_log:bool, decreasing_lr:bool):
    '''
    Experimentations
    '''
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, test_dataset = load_tinygen_image(model_name, tf=Config.TRANSFORM)
    
    print(f"Opération sur {device}")
    print(f"Dataset utilisé '{model_name}' - Classes: {train_dataset.classes}")
    
    dataloader_train = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE_LAB, shuffle=True, num_workers=4, pin_memory=True)
    dataloader_test = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE_LAB, shuffle=False)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    accuracy_list = []
    
    # moyenne sur chaque token
    means = []
    iterations = 8
    
    
    for tokens in Config.ADD_TOKENS_LAB:
        means = []
        
        print("--------------------------------------------------------------------")
        
        for it in range(iterations): 
            model = train_model(model_name, dataloader_train, dataloader_test, tokens, wandb_log, decreasing_lr, device)
            acc = test_model(dataloader_test, device, model)
            means.append(acc)
            
        
            print(f"iteration {it}, {model_name}: acc {means[-1]}")
            
            if wandb_log:
                wandb.log({f"Accuracy/{model_name}": acc, "tokens": tokens})
            accuracy_list.append(acc)
            
        print(f"accuracy mean: {np.mean(np.array(means))}")
    
    if save_image:
        if not os.path.exists('PLOTS'):
            os.makedirs('PLOTS')
        file_name = f'PLOTS/accuracyplot_{timestamp}.png'
        plt.figure(figsize=(11, 11))
        plt.plot(Config.ADD_TOKENS_LAB, accuracy_list, label='Accuracy_token')
        plt.legend()
        plt.grid()
        plt.savefig(file_name)
    return accuracy_list

if __name__ == '__main__':



    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print(f"Date d'entraînement: {timestamp}")
    print("Mode:", "debug" if Config.DEBUG else "run")
    print_verbose(show=False, lab=True)
    
    if Config.WANDB_LOG:
        wandb.init(
            project="Encoder-DecoderProject",
            name=f"AT ALL R{Config.RESIZE_SHAPE} dataset {timestamp}",
            config={"architecture": "dinov2plusllma"}
        )


        wandb.define_metric("tokens")
        wandb.define_metric("Accuracy/*", step_metric="tokens")
        
    
    
    for model in MODEL_NAMES:

        run_experiment(model, Config.SAVE_IMAGE, Config.WANDB_LOG, Config.DECREASING_LR_LAB)
        
    if Config.WANDB_LOG:
        wandb.finish()
        
        
        
        
        
