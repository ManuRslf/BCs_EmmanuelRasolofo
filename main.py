import time
from util import *
from configs import Config
from configs import gaussianTF, jpegTF
import wandb

MODEL_NAMES = ['biggan', 'vqdm', 'sdv5', 'wukong', 'adm', 'glide', 'midjourney']
training_type = ['token', 'cross', 'llama']


def one_by_one(univariee_training:str):


    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print(f"Date d'entraînement: {timestamp}")
    print("Mode:", "debug" if Config.DEBUG else "run")

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

        # metrique pour hiddensize de llama
        wandb.define_metric("llama_hsl")
        wandb.define_metric("Accuracy_hsl/*", step_metric="llama_hsl")

        # metrique pour gaussian noise
        wandb.define_metric("std_gaussian_noise")
        wandb.define_metric("Accuracy_gaussian/*", step_metric="std_gaussian_noise")
        
        
        # metrique pour jpeg compression
        wandb.define_metric("quality")
        wandb.define_metric("Accuracy_jpegcomp/*", step_metric="quality")

    for model in MODEL_NAMES:
        wandb.define_metric(f"Accuracy_cross_model{model}/*", step_metric="tokens")
        
        if univariee_training=='token':
            print_verbose()
            run_experiment_tokens(model, Config.SAVE_IMAGE, Config.WANDB_LOG, Config.DECREASING_LR_LAB)
            
        if univariee_training=='cross':
            print_verbose()
            cross_model(model, Config.WANDB_LOG, Config.DECREASING_LR_LAB)
            
        if univariee_training=='llama':
            print_verbose()
            run_experiment_llama(model, Config.WANDB_LOG, Config.DECREASING_LR_LAB)

        if univariee_training=='llama2':
            
            Config.Adapter_EXTERN = True
            Config.Adapter =True
            
            print_verbose()
            run_experiment_llama2(model, Config.WANDB_LOG, Config.DECREASING_LR_LAB)
            
        if univariee_training=='gaussiannoise':
            print_verbose()
            run_experiment_gaussian(Config.MODEL, Config.WANDB_LOG, Config.DECREASING_LR_LAB)

        if univariee_training=='jpeg':
            print_verbose()
            run_experiment_quality(Config.MODEL, Config.WANDB_LOG, Config.DECREASING_LR_LAB)

    if Config.WANDB_LOG:
        wandb.finish()

def Config_model_run(univariee_training:str=None):

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print(f"Date d'entraînement: {timestamp}")
    print("Mode:", "debug" if Config.DEBUG else "run")

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
        
        
        # metrique pour hiddensize de llama
        wandb.define_metric("llama_hsl")
        wandb.define_metric("Accuracy_hsl/*", step_metric="llama_hsl")
        
        # metrique pour gaussian noise
        wandb.define_metric("std_gaussian_noise")
        wandb.define_metric("Accuracy_gaussian/*", step_metric="std_gaussian_noise")
        
        
        # metrique pour jpeg compression
        wandb.define_metric("quality")
        wandb.define_metric("Accuracy_jpegcomp/*", step_metric="quality")
        
        wandb.define_metric("epoch")
        wandb.define_metric("Train/*", step_metric="epoch")
        
        

    if univariee_training=='token':
        print_verbose()
        run_experiment_tokens(Config.MODEL, Config.SAVE_IMAGE, Config.WANDB_LOG, Config.DECREASING_LR_LAB)
        
    if univariee_training=='cross':
        print_verbose()
        cross_model(Config.MODEL, Config.WANDB_LOG, Config.DECREASING_LR_LAB)
        
    if univariee_training=='llama':
        print_verbose()
        run_experiment_llama(Config.MODEL, Config.WANDB_LOG, Config.DECREASING_LR_LAB)
        
    if univariee_training=='llama2':
        print_verbose()
        Config.Adapter_EXTERN = True
        Config.Adapter =True
        run_experiment_llama2(Config.MODEL, Config.WANDB_LOG, Config.DECREASING_LR_LAB)
    
    if univariee_training=='gaussiannoise':
        print_verbose()
        run_experiment_gaussian(Config.MODEL, Config.WANDB_LOG, Config.DECREASING_LR_LAB)

    if univariee_training=='jpeg':
        print_verbose()
        run_experiment_quality(Config.MODEL, Config.WANDB_LOG, Config.DECREASING_LR_LAB)
         
    elif univariee_training is None:
        """
        entrainement selon l'epoch. Comparasion avec cosineal et normal
        """        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        _ = simple_training(Config.MODEL, 16, False, device)
        _ = simple_training(Config.MODEL, 16, True, device)

    
    if Config.WANDB_LOG:
        wandb.finish()
        

def IFDEBUG():
    print("\033[92m \n\n\nTest fonction..\n\n \033[0m")
    '''
    Config_model_run('token')
    print("\033[92mToken OK..\033[0m")

    Config_model_run('cross')
    print("\033[92mCross OK..\033[0m")

    Config_model_run('llama')
    print("\033[92mLLAMA OK..\033[0m")

    Config_model_run('llama2')
    print("\033[92mLLAMA2 OK..\033[0m")

    Config_model_run('gaussiannoise')
    print("\033[92mGAUSSIAN OK..\033[0m")
    '''
    Config_model_run('jpeg')
    print("\033[92mJPEG OK..\033[0m")

    print('\n OK.')


def train():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print(f"Date d'entraînement: {timestamp}")
    print("Mode:", "debug" if Config.DEBUG else "run")
    print_verbose()

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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    _ = simple_training('midjourney', 5, True, device)
    
    if Config.WANDB_LOG:
        wandb.finish()
    

        
if __name__ == '__main__' and not Config.DEBUG:
    #one_by_one('llama') 
    print("ENTRAINEMENT EN ENTRAINANT SUR DES DONNEéS DEGRADéES")
    Config_model_run('gaussiannoise')
    Config_model_run('jpeg')
    '''
    from copy import deepcopy
    T = deepcopy(Config.TRANSFORM)
    Config.TRANSFORM = jpegTF(quality=15)  
    Config_model_run('jpeg')
    Config.TRANSFORM = T
    Config.DECREASING_LR_LAB=False
    Config_model_run('llama2')
'''
    #train()
    
    
if __name__ == '__main__' and Config.DEBUG:
    IFDEBUG()
