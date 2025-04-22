import time
from util import *
from configs import Config
import wandb

MODEL_NAMES = ['biggan', 'vqdm', 'sdv5', 'wukong', 'adm', 'glide', 'midjourney']
training_type = ['token', 'cross', 'llama']


def one_by_one(univariee_training:str):


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
        wandb.define_metric(f"Accuracy_cross_model{Config.MODEL}/*", step_metric="tokens")

        # metrique pour num hidden layer de llama
        wandb.define_metric("llama_nhl")
        wandb.define_metric("Accuracy_lnhl/*", step_metric="llama_nhl")




    for model in MODEL_NAMES:
        wandb.define_metric(f"Accuracy_cross_model{model}/*", step_metric="tokens")
        
        if univariee_training=='token':
            print("training en variant le nombre de token")
            run_experiment(model, Config.SAVE_IMAGE, Config.WANDB_LOG, Config.DECREASING_LR_LAB)
            
        if univariee_training=='cross':
            print("training en test generalisée")
            cross_model(model, Config.WANDB_LOG, Config.DECREASING_LR_LAB)
            
        if univariee_training=='llama':
            print("training en variant le nombre de layer de llama")
            run_experiment_llama(model, Config.WANDB_LOG, Config.DECREASING_LR_LAB)
    if Config.WANDB_LOG:
        wandb.finish()

def Config_model_run(univariee_training:str=None):

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print(f"Date d'entraînement: {timestamp}")
    print("Mode:", "debug" if Config.DEBUG else "run")
    print_verbose(show=False, lab=True)

    if Config.WANDB_LOG:
        wandb.init(
            project="Encoder-DecoderProject",
            name=f"AT {Config.MODEL if Config.MODEL is not None else "Merged"} R{Config.RESIZE_SHAPE} dataset {timestamp}",
            config={"architecture": "dinov2plusllma"}
        )

        #definintion des metriques pour wandb
        wandb.define_metric("tokens")
        wandb.define_metric("Accuracy/*", step_metric="tokens")
        wandb.define_metric(f"Accuracy_cross_model{Config.MODEL}/*", step_metric="tokens")
        # metrique pour num hidden layer de llama       
        wandb.define_metric("llama_nhl")
        wandb.define_metric("Accuracy_lnhl/*", step_metric="llama_nhl")


    if univariee_training=='token':
        run_experiment(Config.MODEL, Config.SAVE_IMAGE, Config.WANDB_LOG, Config.DECREASING_LR_LAB)
        
    if univariee_training=='cross':
        cross_model(Config.MODEL, Config.WANDB_LOG, Config.DECREASING_LR_LAB)
        
    if univariee_training=='llama':
        run_experiment_llama(Config.MODEL, Config.WANDB_LOG, Config.DECREASING_LR_LAB)
        
    else:
        """
        entrainement selon l'epoch. Comparasion avec cosineal et normal
        """        
        wandb.define_metric("epoch")
        wandb.define_metric("Train/*", step_metric="epoch")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        _ = simple_training(Config.MODEL, 16, False, device)
        _ = simple_training(Config.MODEL, 16, True, device)

    
    if Config.WANDB_LOG:
        wandb.finish()
        



def train_and_visu():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    _ = simple_training('midjourney', 5, True, device)
    

        
if __name__ == '__main__':
    #one_by_one('llama')    
    Config_model_run()
    #train_and_visu()
