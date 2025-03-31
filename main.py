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
        


    for model in MODEL_NAMES:

        run_experiment(model, Config.SAVE_IMAGE, Config.WANDB_LOG, Config.DECREASING_LR_LAB)
        
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
            name=f"AT ALL R{Config.RESIZE_SHAPE} dataset {timestamp}",
            config={"architecture": "dinov2plusllma"}
        )


        wandb.define_metric("tokens")
        wandb.define_metric("Accuracy/*", step_metric="tokens")
        

    run_experiment(Config.MODEL, Config.SAVE_IMAGE, Config.WANDB_LOG, Config.DECREASING_LR_LAB)
        
    if Config.WANDB_LOG:
        wandb.finish()