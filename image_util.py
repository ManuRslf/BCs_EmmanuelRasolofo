import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from configs import Config


MODEL_NAMES = ['biggan', 'vqdm', 'sdv5', 'wukong', 'adm', 'glide', 'midjourney']

def load_tinygen_image(model:str=None, tf:transforms.Compose=None):
    '''
    Charge le dataset TinygenImage selon le modèle sélectionné
    '''
    path_to_data = [
        r'DATA/tinygenimage/imagenet_ai_0419_biggan', 
        r'DATA/tinygenimage/imagenet_ai_0419_vqdm', 
        r'DATA/tinygenimage/imagenet_ai_0424_sdv5',
        r'DATA/tinygenimage/imagenet_ai_0424_wukong',
        r'DATA/tinygenimage/imagenet_ai_0508_adm',
        r'DATA/tinygenimage/imagenet_glide',
        r'DATA/tinygenimage/imagenet_midjourney'
    ]
    valid_models = {MODEL_NAMES[i]: path_to_data[i] for i in range(len(MODEL_NAMES))}
    
    if model is not None and model not in MODEL_NAMES:
        raise ValueError(f"Choix invalide -> {list(valid_models.keys())}")
    
    base_path = os.getcwd()
    if model is None:
        merged_path = 'DATA/tinygenimage_merged'
        train_path = os.path.join(base_path, merged_path, 'train').replace("\\", "/")
        test_path = os.path.join(base_path, merged_path, 'test').replace("\\", "/")
    else:
        train_path = os.path.join(base_path, valid_models[model], 'train').replace("\\", "/")
        test_path = os.path.join(base_path, valid_models[model], 'test').replace("\\", "/")
    
    return ImageFolder(root=train_path, transform=tf), ImageFolder(root=test_path, transform=tf)

def print_verbose(show:bool=True, lab:bool=False, *args):
    '''
    Affiche les informations
    '''
    if show:
        print(f"Images redimensionnées en {Config.RESIZE_SHAPE}x{Config.RESIZE_SHAPE}")
        print(f"Tokens additionnels: {Config.ADD_TOKENS}")
        print(f"LLMA: {Config.NUM_HIDDEN_LAYER_LLMA} couches, taille {Config.HIDDEN_SIZE}")
        print(f"Batch size: {Config.BATCH_SIZE}, LR: {Config.LR}, Époques: {Config.EPOCHS}")
        return
    if lab:
        print(f"Images redimensionnées en {Config.RESIZE_SHAPE}x{Config.RESIZE_SHAPE}")
        print(f"Tokens additionnels (lab): {Config.ADD_TOKENS_LAB}")
        print(f"LLMA (lab): {Config.NUM_HIDDEN_LAYER_LLMA_LAB} couches, taille {Config.HIDDEN_SIZE_LAB}")
        print(f"Batch size (lab): {Config.BATCH_SIZE_LAB}, LR (lab): {Config.LR_LAB}, Époques (lab): {Config.EPOCHS_LAB}")
        print(f"LLAMA num hidden : {Config.NHL_LAB}")
        return
        
    else:    
        for ar in args:
            print(str(ar))
        