"""
CE FICHIER CONTIENT ESSENTIELLEMENT L'INTEGRALITé DES FONCTIONS POUR L'ENTRAINEMENT UNIVARIé
COMME LA VARIATION DES NOMBRES DE TOKENS AJOUTé, LE NOMBRE DE COUCHE CACHé DE LLAMA,..
"""


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
from configs import gaussianTF
from configs import jpegTF

MODEL_NAMES = ['biggan', 'vqdm', 'sdv5', 'wukong', 'adm', 'glide', 'midjourney']

def train_model(model_name:str,
                dataloader_train:DataLoader,
                additional_tokens:int,
                wandb_log:bool,
                decreasing_lr:bool,
                device: torch.device,
                tsne:bool=False,
                dataloader_test=None):
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
    
    if tsne:
        # visualisation du dernier token qui est initialisé aleatoirement
        model.visualize_emb_class(model_name, dataloader_test, device, -1)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

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
        
        # log tsne representation dans les dossiers
        if tsne and epoch % 5 == 0:
            model.visualize_emb_class(model_name, dataloader_test, device, epoch)
        
        if wandb_log:
            
            #wandb.log({f"Train/Loss{timestamp}": avg_loss, "epoch": epoch})
            pass
        
        #decommenter pour voir loss
        print(f"Epoch {epoch} - Loss moyenne: {avg_loss}")
    print("Entraînement terminé.")
    return model



def train_model_llama_params(model_name:str,
                dataloader_train:DataLoader,
                num_hidden_layer:int,
                hidden_size:int,
                wandb_log:bool,
                decreasing_lr:bool,
                device: torch.device,
                tsne:bool=False,
                dataloader_test=None,
                EPOCHS:int=Config.EPOCHS_LAB
                ):
    '''
    Fonction d'entrainement pour les paramètres donnés    
    '''
    print(f"[{model_name}]Entraînement avec {num_hidden_layer} couche de llama et de tailles {hidden_size} pour chaque couches")
    
    llama_config = LlamaConfig(num_hidden_layers=num_hidden_layer, 
                               hidden_size=hidden_size)
    model = CustomClassifier.CustomClassifier(
        llama_config, 
        dinov2_name=Config.DINOV2_NAME, 
        hidden_size=hidden_size,
        additional_tokens=Config.add_tokens_lab
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    
    if decreasing_lr:
        optimizer = AdamW(model.parameters(), lr=Config.LR_LAB)
        scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS_LAB * len(dataloader_train))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR_LAB)
        scheduler = None
    
    print("TRAINING...")
    
    if tsne:
        # visualisation du dernier token qui est initialisé aleatoirement
        model.visualize_emb_class(dataloader_test, device, -1)

    for epoch in range(EPOCHS):
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
        
        # log tsne representation dans les dossiers
        if tsne and epoch % 5 == 0:
            model.visualize_emb_class(dataloader_test, device, epoch)
        
        if wandb_log:
            # wandb.log({"Train/Loss": avg_loss, "epoch": epoch})
            pass
        
        #decommenter pour voir loss
        print(f"Epoch {epoch} - Loss moyenne: {avg_loss}")
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

def simple_training(model_name:str, additional_token:int, decreasing_lr:bool, device):
    
    '''
    une methode qui effectue un entrainement simple. Utilisé pour visualisation du dernuer token passé apres llama
    '''
    
    train_dataset, test_dataset = load_tinygen_image(model_name, tf=Config.TRANSFORM)
    
    print(f"Opération sur {device}")
    print(f"Dataset utilisé '{model_name}' - Classes: {train_dataset.classes}")
    '''
    print_verbose(False, False, f"Resize shape : {Config.RESIZE_SHAPE}",
                                f"Tokens additionels : {additional_token}",
                                f"LLaMA hidden size et num layer : {Config.NUM_HIDDEN_LAYER_LLMA_LAB}, {Config.HIDDEN_SIZE_LAB}",
                                f"DINOv2 model : {Config.DINOV2_NAME}")
    '''
    dataloader_train = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE_LAB, shuffle=True, num_workers=4, pin_memory=True)
    dataloader_test = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE_LAB, shuffle=False)
    
    model_trained = train_model(model_name,
                dataloader_train,
                additional_token,
                wandb_log=Config.WANDB_LOG,
                decreasing_lr=decreasing_lr,
                device=device,
                tsne=Config.TSNE_LOG,
                dataloader_test=dataloader_test)
    
    return model_trained

def run_experiment_tokens(model_name:str, save_image:bool, wandb_log:bool, decreasing_lr:bool):
    '''
    Experimentations: variation du nombre de tokens additionels
    '''
    print("\033[93m \n\nENTRAINEMENT: VARIAtION DU NOMBRE DE TOKENS\n\n \033[0m")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, test_dataset = load_tinygen_image(model_name, tf=Config.TRANSFORM)
    
    print(f"Opération sur {device}")
    print(f"Dataset utilisé '{model_name}' - Classes: {train_dataset.classes}")
    
    dataloader_train = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE_LAB, shuffle=True, num_workers=4, pin_memory=True)
    dataloader_test = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE_LAB, shuffle=False)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    accuracy_list = []
    
    # on fait la moyenne pour k iterations
    means = []
    iterations = Config.ITERATION
    
    
    for tokens in Config.ADD_TOKENS_LAB:
        means = []
        
        print("-" * 100)
        
        if iterations > 1:
            for it in range(iterations): 
                model = train_model(model_name, dataloader_train, tokens, wandb_log, decreasing_lr, device)
                acc = test_model(dataloader_test, device, model)
                means.append(acc)
                
            
                print(f"iteration {it}, {model_name}: acc {means[-1]}")
            if wandb_log:
                wandb.log({f"Accuracy/{model_name}": np.mean(np.array(means)), "tokens": tokens})
            print(f"accuracy mean: {np.mean(np.array(means))}")

                
        else:
            model = train_model(model_name, dataloader_train, tokens, wandb_log, decreasing_lr, device)
            acc = test_model(dataloader_test, device, model)
            
            if wandb_log:
                wandb.log({f"Accuracy/{model_name}": acc, "tokens": tokens})
            accuracy_list.append(acc)
            print(f"accuracy mean: {acc}")        
    
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

def test_other_dataset(device:torch.device, model:nn.Module, verbose:bool = True):
    """
    Test le model par rapport à un dataset
    """
    model.eval()
    predictions = []
    true_labels = []
    
    MODEL_NAMES = ['biggan', 'vqdm', 'sdv5', 'wukong', 'adm', 'glide', 'midjourney']
    
    # tous les dataloaders de chaque datasets
    model_dataloaders = {
        i : load_tinygen_image(i, tf=Config.TRANSFORM)[1] for i in MODEL_NAMES
    }
    accuracy_result = dict.fromkeys(MODEL_NAMES, None)
    
    for name, dataset in model_dataloaders.items():
        predictions = []
        true_labels = []
        
        dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE_LAB, shuffle=False)
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                
        acc = accuracy_score(true_labels, predictions)
        
        accuracy_result[name] = acc
        
        if verbose:
            print(f"Accuracy : {acc} avec le dataset --------------------> {name}")
            print("Rapport de classification :")
            print(classification_report(true_labels, predictions, target_names=['ia', 'nature'], zero_division=1))
            print("-" * 100)
    return accuracy_result

def cross_model(model_name:str, wandb_log:bool, decreasing_lr:bool):
    '''
    Training: une dataset en particulier, Test: evalu sa performance par rapport au autre modéle
    '''
    print("\033[93m \n\nENTRAINEMENT: CROSS GENRATOR IMAGE CLASSIFICATION\n\n \033[0m")

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # dataloaders du model actuel
    train_dataset, _ = load_tinygen_image(model_name, tf=Config.TRANSFORM)
    
    print(f"Opération sur {device}")
    print(f"Dataset utilisé '{model_name}' - Classes: {train_dataset.classes} -> perf_inter_model")
    
    dataloader_train = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE_LAB, shuffle=True, num_workers=4, pin_memory=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    
    for tokens in Config.ADD_TOKENS_LAB_perf:      
          
        print("-" * 100)
        
        model = train_model(model_name, dataloader_train, tokens, wandb_log, decreasing_lr, device)
        
        ACCs = test_other_dataset(device=device, model=model)
        
        
        for name, accuracy in ACCs.items():
            
            print(f"Accuracy {model_name} avec le dataset {name} = {accuracy}")
            
            if wandb_log:
                wandb.log({f"Accuracy_cross_model{model_name}/{name}": accuracy, "tokens": tokens})
        
def run_experiment_llama(model_name:str, wandb_log:bool, decreasing_lr:bool):
    '''
    Experimentations: entrainement avec hyperparametre univarié sur la variation des parametre de llama 
    tel que le nombres de couche caché
    '''
    print("\033[93m \n\nENTRAINEMENT: VARIAtION DU NOMBRE DE COUCHES CACHEES DE LLAMA\n\n \033[0m")

    print("\n\nCECI EST UN ENTRAINEMENT EN VARIANT LE NOMBRE DE COUCHES CACHEES DE LLAMA\n\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, test_dataset = load_tinygen_image(model_name, tf=Config.TRANSFORM)
    
    print(f"Opération sur {device}")
    print(f"Dataset utilisé '{model_name}' - Classes: {train_dataset.classes}")
    
    dataloader_train = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE_LAB, shuffle=True, num_workers=4, pin_memory=True)
    dataloader_test = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE_LAB, shuffle=False)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    accuracy_list = []
    
    # on fait la moyenne pour k iterations
    means = []
    iterations = Config.ITERATION
    
    for num_hidden_layer in Config.NHL_LAB:
        means = []
        
        print("-" * 100)
        
        if iterations > 1:
            for it in range(iterations): 
                # on procède à l'entrainement et on recupère le modèle entrainé qu'on fait tester dans les validations sets
                model = train_model_llama_params(model_name, 
                                                 dataloader_train, 
                                                 num_hidden_layer, 
                                                 Config.HIDDEN_SIZE_LAB, 
                                                 wandb_log, 
                                                 decreasing_lr, 
                                                 device)
                acc = test_model(dataloader_test, device, model)
                means.append(acc)
                
            
                print(f"iteration {it}, {model_name}: acc {means[-1]}")
            if wandb_log:
                wandb.log({f"Accuracy_lnhl/{model_name}": np.mean(np.array(means)), "llama_nhl": num_hidden_layer})
            print(f"accuracy mean: {np.mean(np.array(means))}")

                
        else:
            model = train_model_llama_params(model_name, 
                                             dataloader_train, 
                                             num_hidden_layer, 
                                             Config.HIDDEN_SIZE_LAB, 
                                             wandb_log, 
                                             decreasing_lr, 
                                             device)
            acc = test_model(dataloader_test, device, model)
            
            if wandb_log:
                wandb.log({f"Accuracy_lnhl/{model_name}": acc, "llama_nhl": num_hidden_layer})
            accuracy_list.append(acc)
            print(f"accuracy mean: {acc}")
    
    return accuracy_list

def run_experiment_llama2(model_name:str, wandb_log:bool, decreasing_lr:bool):
    '''
    Experimentations2: entrainement avec hyperparametre univarié sur la variation des parametre de llama 
    tel que la taille des couches cachées
    '''
    print("\033[93m \n\nENTRAINEMENT: VARIAtION LA TAILLE DE COUCHES CACHEES DE LLAMA\n\n \033[0m")

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
    iterations = Config.ITERATION
    
    for hidden_size, e in zip(Config.HSL_LAB, Config.EPOCHS_HSL):
        means = []
        
        print("-" * 100)
        
        if iterations > 1:
            for it in range(iterations): 
                # on procède à l'entrainement et on recupère le modèle entrainé qu'on fait tester dans les validations sets
                model = train_model_llama_params(model_name, 
                                                 dataloader_train, 
                                                 Config.NUM_HIDDEN_LAYER_LLMA_LAB, 
                                                 hidden_size, 
                                                 wandb_log, 
                                                 decreasing_lr, 
                                                 device,
                                                 EPOCHS=e)
                acc = test_model(dataloader_test, device, model)
                means.append(acc)
                
            
                print(f"iteration {it}, {model_name}: acc {means[-1]}")
            if wandb_log:
                wandb.log({f"Accuracy_hsl/{model_name}": np.mean(np.array(means)), "llama_hsl": hidden_size})
            print(f"accuracy mean: {np.mean(np.array(means))}")

                
        else:
            model = train_model_llama_params(model_name, 
                                             dataloader_train, 
                                             Config.NUM_HIDDEN_LAYER_LLMA_LAB, 
                                             hidden_size, 
                                             wandb_log, 
                                             decreasing_lr, 
                                             device)
            acc = test_model(dataloader_test, device, model)
            
            if wandb_log:
                wandb.log({f"Accuracy_hsl/{model_name}": acc, "llama_hsl": hidden_size})
            accuracy_list.append(acc)
            print(f"accuracy mean: {acc}")
    
    return accuracy_list


def run_experiment_gaussian(model_name:str, wandb_log:bool, decreasing_lr:bool):
    '''
    Experimentations: ajout de bruits dans les données tests
    '''
    print("\033[93m \n\nENTRAINEMENT: AJOUT DE BRUIT GAUSSIEN\n\n \033[0m")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = load_tinygen_image(model_name, tf=Config.TRANSFORM, get='train')
    

    print(f"Opération sur {device}")
    print(f"Dataset utilisé '{model_name}' - Classes: {train_dataset.classes}")
    
    dataloader_train = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE_LAB, shuffle=True, num_workers=4, pin_memory=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    accuracy_list = []
    

    print("-" * 100)
    
    model = train_model(model_name, dataloader_train, Config.add_tokens_lab, wandb_log, decreasing_lr, device)
    print("Test...")
    for std in Config.STD_GAUSSIAN_NOISE:
        print(f"STD -> {std}")
        
        TF = gaussianTF(mean=0, std=std)
        
        test_dataset = load_tinygen_image(model_name, tf=TF, get='test')
        dataloader_test = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE_LAB, shuffle=False)

        acc = test_model(dataloader_test, device, model)        
        if wandb_log:
            wandb.log({f"Accuracy_gaussian/{model_name}{timestamp}": acc, "std_gaussian_noise": std})
        accuracy_list.append(acc)
        
        print(f"accuracy : {acc}")  
              
    return accuracy_list

def run_experiment_quality(model_name:str, wandb_log:bool, decreasing_lr:bool):
    '''
    Experimentations: effet sur l'accuracy par rapport à la dégradation de l'image dans les test
    '''
    print("\033[93m \n\nENTRAINEMENT: DEGRADATION DE LA QUALITé\n\n \033[0m")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = load_tinygen_image(model_name, tf=Config.TRANSFORM, get='train')
    

    print(f"Opération sur {device}")
    print(f"Dataset utilisé '{model_name}' - Classes: {train_dataset.classes}")
    
    dataloader_train = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE_LAB, shuffle=True, num_workers=4, pin_memory=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    accuracy_list = []
    

    print("-" * 100)
    
    model = train_model(model_name, dataloader_train, Config.add_tokens_lab, wandb_log, decreasing_lr, device)
    print("Test...")
    for quality in Config.QUALITY_JPEG_COMPRESSION:
        print(f"Quality -> {quality}")
        
        TF = jpegTF(quality=quality)
        
        test_dataset = load_tinygen_image(model_name, tf=TF, get='test')
        dataloader_test = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE_LAB, shuffle=False)

        acc = test_model(dataloader_test, device, model)        
        if wandb_log:
            wandb.log({f"Accuracy_jpegcomp/{model_name}{timestamp}": acc, "quality": quality})
        accuracy_list.append(acc)
        
        print(f"accuracy : {acc}")  
              
    return accuracy_list


if __name__ == '__main__':



    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print(f"Date d'entraînement: {timestamp}")
    print("Mode:", "debug" if Config.DEBUG else "run")
    print_verbose()
    
    if Config.WANDB_LOG:
        wandb.init(
            project="Encoder-DecoderProject",
            name=f"AT ALL R{Config.RESIZE_SHAPE} dataset {timestamp}",
            config={"architecture": "dinov2plusllma"}
        )


        wandb.define_metric("tokens")
        wandb.define_metric("Accuracy/*", step_metric="tokens")
        
    
    
    for model in MODEL_NAMES:

        run_experiment_tokens(model, Config.SAVE_IMAGE, Config.WANDB_LOG, Config.DECREASING_LR_LAB)
        
    if Config.WANDB_LOG:
        wandb.finish()
        
        
        
        
        
