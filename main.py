import time
from util import *
from configs import Config
import wandb

MODEL_NAMES = ['biggan', 'vqdm', 'sdv5', 'wukong', 'adm', 'glide', 'midjourney']


def one_by_one():

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



    for model in MODEL_NAMES:
        wandb.define_metric(f"Accuracy_cross_model{model}/*", step_metric="tokens")
        #run_experiment(model, Config.SAVE_IMAGE, Config.WANDB_LOG, Config.DECREASING_LR_LAB)
        cross_model(model, Config.WANDB_LOG, Config.DECREASING_LR_LAB)
        
    if Config.WANDB_LOG:
        wandb.finish()

def Config_model_run():

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


        wandb.define_metric("tokens")
        wandb.define_metric("Accuracy/*", step_metric="tokens")
        wandb.define_metric(f"Accuracy_cross_model{Config.MODEL}/*", step_metric="tokens")


    cross_model(Config.MODEL, Config.WANDB_LOG, Config.DECREASING_LR_LAB)
        
    if Config.WANDB_LOG:
        wandb.finish()
        
        
if __name__ == '__main__':
    one_by_one()
    #Config_model_run()