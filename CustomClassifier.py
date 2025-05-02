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
      -Une couche linéaire pour adapter la sortie de DinoV2 dans LLaMA
      -Un modèle LLaMA pour decoder
      -Une couche linéaire de classification pour la sortie (classification binaire)
    '''
    def __init__(self, 
                 llama_config:LlamaConfig, 
                 dinov2_name:str='facebook/dinov2-base', 
                 hidden_size:int=Config.HIDDEN_SIZE_LAB,
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
        self.hidden_size = hidden_size
        #DinoV2 figé
        self.dinov2_model = AutoModel.from_pretrained(dinov2_name)
        self.dinov2_model.requires_grad_(False)
        self.dinov2_model.eval()
        self.num_token_additional = additional_tokens
        
        # si la dimension de LLaMA est differente de celui de Dinov2, on adapte la dimension
        if Config.Adapter:
            self.Adapter = nn.Linear(Config.Dinov2_token_dim[Config.DINOV2_NAME], hidden_size)
            
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
        
        # on adapte
        if Config.Adapter:
            features = self.Adapter(features)
            
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

        if Config.Adapter:
            features = self.Adapter(features)

        batch_size = features.size(0)
        add_tokens_expanded = self.add_tokens.unsqueeze(0).repeat(batch_size, 1, 1)
        features = torch.cat((features, add_tokens_expanded), dim=1)
        
        #REFLEXION
        outputs = self.llama(inputs_embeds=features)
        logits = outputs['logits'][:, -1] 
            
        with torch.no_grad():
            l=logits.cpu().detach().numpy()  
        return l  

    def visualize_emb_class(self, model_name, dataloader, device, epoch:int):
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
        file_name = f'tsne_log/tsne_{epoch}_{timestamp}{model_name}.png'

        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df, x="x", y="y", hue="label", palette="deep", s=60)
        plt.title(f"Token additionel LLAMA avec t-SNE ({model_name})")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)
        plt.savefig(file_name)  
 
          
    

