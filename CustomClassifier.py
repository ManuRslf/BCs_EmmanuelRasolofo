import torch
from torch import nn
from transformers import AutoModel, LlamaForCausalLM, LlamaConfig
from sklearn.metrics import accuracy_score, classification_report
from image_util import load_tinygen_image, print_verbose
from torch.utils.data import DataLoader
from configs import Config
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import numpy as np
import os
import time
class CustomClassifier(nn.Module):
    '''
    Modèle personnalisé qui combine :
      -Un extracteur de caractéristique DinoV2 
      -Un modèle LLaMA pour decoder
      -Une couche linéaire de classification pour la sortie (classification binaire)
    '''
    def __init__(self, 
                 llama_config:LlamaConfig, 
                 dinov2_name:str='facebook/dinov2-base', 
                 hidden_size:int=768,
                 additional_tokens:int=2):
        '''
        arguments:
            llama_config (LlamaConfig): Configuration du modèle LLaMA
            dinov2_name (str): Nom du modèle pré-entraîné DinoV2
            hidden_size (int): Taille cachée du modèle
            additional_tokens (int): Nombre de tokens additionnels à optimiser
        '''
        super().__init__()
        
        #tokens additionels
        self.add_tokens = nn.Parameter(torch.randn(additional_tokens, llama_config.hidden_size))
        
        #DinoV2 figé
        self.dinov2_model = AutoModel.from_pretrained(dinov2_name)
        self.dinov2_model.requires_grad_(False)
        self.dinov2_model.eval()
        
        # LLaMA sans tête LM
        self.llama = LlamaForCausalLM(llama_config)
        self.llama.lm_head = nn.Identity()
        
        self.classifier = nn.Linear(llama_config.hidden_size, 2)
        
    def forward(self, x:torch.Tensor):
        '''
        Forward pass

        x (torch.Tensor) -> torch.Tensor: logits de classification
        '''
        #REPRESENTATION
        with torch.no_grad():
            features = self.dinov2_model(x)['last_hidden_state']
        

        batch_size = features.size(0)
        add_tokens_expanded = self.add_tokens.unsqueeze(0).repeat(batch_size, 1, 1)
        features = torch.cat((features, add_tokens_expanded), dim=1)
        
        #REFLEXION
        outputs = self.llama(inputs_embeds=features)
        logits = outputs['logits'][:, -1] 
        logits = self.classifier(logits)
        return logits
    
    # extraction embedding
    def ext_embedding(self, x):   
        #REPRESENTATION
        with torch.no_grad():
            features = self.dinov2_model(x)['last_hidden_state']
        

        batch_size = features.size(0)
        add_tokens_expanded = self.add_tokens.unsqueeze(0).repeat(batch_size, 1, 1)
        features = torch.cat((features, add_tokens_expanded), dim=1)
        
        #REFLEXION
        outputs = self.llama(inputs_embeds=features)
        logits = outputs['logits'][:, -1] 
            
        with torch.no_grad():
            l=logits.cpu().detach().numpy()  
        return l  

    def visualize_emb_class(self, dataloader, device, epoch:int):
        self.dinov2_model.eval()
        all_embeddings = []
        all_labels = []

        for images, labels in dataloader:
            images = images.to(device)
            emb = self.ext_embedding(images)  
            all_embeddings.append(emb)
            
            all_labels.extend([label.item() if isinstance(label, torch.Tensor) else label for label in labels])

        # N, hidden_dim
        embeddings = np.concatenate(all_embeddings, axis=0)

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        reduced = tsne.fit_transform(embeddings)

        #dataframe
        df = pd.DataFrame(reduced, columns=["x", "y"])
        df["label"] = all_labels
        
        # sauvergarde de l'image dans le dossier
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if not os.path.exists('tsne_log'):
            os.makedirs('tsne_log')
        file_name = f'tsne_log/tsne_{epoch}_{timestamp}.png'

        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df, x="x", y="y", hue="label", palette="deep", s=60)
        plt.title("Last token LLAMA avec t-SNE")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True)
        plt.savefig(file_name)    
          
    

def training_testing(model_name:str=None):
    '''Fonction d'entraînement et d'évaluation sur le dataset spécifié'''
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset_train, dataset_test = load_tinygen_image(model_name, tf=Config.TRANSFORM)
    
    print(f"Opération sur {device}")
    print(f"Dataset utilisé '{model_name}' - Classes: {dataset_train.classes}")
    print_verbose(show=Config.SHOW_INFO)
    
    #DataLoaders
    dataloader_train = DataLoader(dataset_train, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    #Configuration LLaMA
    llama_config = LlamaConfig(num_hidden_layers=Config.NUM_HIDDEN_LAYER_LLMA, 
                               hidden_size=Config.HIDDEN_SIZE)
    
    model = CustomClassifier(llama_config, additional_tokens=Config.ADD_TOKENS).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    
    print("TRAINING...")
    for epoch in range(Config.EPOCHS):
        total_loss = 0.0
        count = 0
        for inputs, labels in dataloader_train:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            count += 1
        print(f"Epoch {epoch} - Loss moyenne: {total_loss/count:.2f}")
    
    print("Entraînement terminé.")
    
    # Test
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
    print(f"Accuracy : {acc}")
    print("Rapport de classification :")
    print(classification_report(true_labels, predictions, target_names=['ia', 'nature']))
    
if __name__ == '__main__':
    #training_testing(Config.MODEL)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset_train, dataset_test = load_tinygen_image('midjourney', tf=Config.TRANSFORM)
    dataloader_train = DataLoader(dataset_train, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    llama_config = LlamaConfig(num_hidden_layers=Config.NUM_HIDDEN_LAYER_LLMA, 
                               hidden_size=Config.HIDDEN_SIZE)
    
    model = CustomClassifier(llama_config, additional_tokens=Config.ADD_TOKENS).to(device)
    
    model.visualize_emb_class(dataloader_train, device)