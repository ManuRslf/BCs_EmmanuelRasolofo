import torch 
from torch import nn
from utils import *
import configurations
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import AutoModel, LlamaForCausalLM, LlamaConfig
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


class Custom_Classifier(nn.Module):
    def __init__(self, llma_config, dinov2_name:str='facebook/dinov2-base', additional_token:int=2):
        super().__init__()
        
        #gen random tokens, to optimize during training
        self.ADD_TOK = nn.Parameter(torch.randn(additional_token, llma_config.hidden_size))
        
        #instanciate a pretrained dinov2 model
        self.DINOV2_MODEL = AutoModel.from_pretrained(dinov2_name)
        self.DINOV2_MODEL.requires_grad_(False)
        self.DINOV2_MODEL.eval()
        
        self.LLMA = LlamaForCausalLM(llma_config)
        self.LLMA.lm_head = nn.Identity()
        
        self.CLASSIFIER = nn.Linear(llma_config.hidden_size, 2)
        
        self.proj = nn.Linear(384, configurations.HIDDEN_SIZE)
        
    def forward(self, x):
        
        #tokenize through dinov2 model
        
        ## quand gradient dinov2 bloqué
        with torch.no_grad():
          x = self.DINOV2_MODEL(x)['last_hidden_state']
        

        #add additional reasoning tokens  (x, y) -> 1,x,y -> batchsize, x, y
        x = torch.cat((x, self.ADD_TOK.unsqueeze(0).repeat(x.size(0), 1, 1)), dim = 1)
        
        #token -> llmamodel
        x = self.LLMA(inputs_embeds=x)
        
        x = x['logits'][:, -1]
        x = self.CLASSIFIER(x)
        
        return x
    
    
def training_testing(model_i:str=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #chargement des données
    dataset_train, dataset_test = TinygenImage(model_i, tf=configurations.tf)
    
    
    print(f"Operation on {device}")
    print(f"Using {model_i} DATASET, Classes in dataset: {dataset_train.classes}")
    verbose(configurations.show_info)
    
    
    dataloader_train = DataLoader(dataset_train, 
                                  batch_size=configurations.BATCH_SIZE, 
                                  shuffle=True, 
                                  num_workers=4, 
                                  pin_memory=True)


    dataloader_test = DataLoader(dataset_test,
                                 batch_size=configurations.BATCH_SIZE,
                                 shuffle=False)


    llama_config = LlamaConfig(num_hidden_layers=configurations.NUM_HIDDEN_LAYER_LLMA, hidden_size=configurations.HIDDEN_SIZE)
    
    model = Custom_Classifier(llama_config, additional_token=configurations.ADD_TOKENS).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=configurations.LR)

    #initial_dinov2_w = model.DINOV2_MODEL.encoder.layer[0].mlp.fc1.weight[:3, :3]


    #Training
    print("Training...")
    
    for e in range(configurations.EPOCHS):
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
    print(f"Accuracy : {ACC}\nClassification report:\n{classification_report(Abatch_labels, Abatch_predictions, target_names=['ia', 'nature'])}")
    
    
if __name__ == '__main__':
    training_testing(configurations.MODEL)
    