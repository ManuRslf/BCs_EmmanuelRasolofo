import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from configs import Config
from configs import gaussianTF
from configs import jpegTF
import torch
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np

MODEL_NAMES = ['biggan', 'vqdm', 'sdv5', 'wukong', 'adm', 'glide', 'midjourney']

def load_tinygen_image(model:str=None, tf:transforms.Compose=None, get:str='all'):
    '''
    Charge le dataset TinygenImage selon le modèle sélectionné
    '''
    get_str = ['all', 'train', 'test']

    if get not in get_str: raise ValueError("Choisissez bien les datasets souhaité. ", get_str)
    
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
    

    if get == 'all':
        return ImageFolder(root=train_path, transform=tf), ImageFolder(root=test_path, transform=tf)
    if get == 'train':
        return ImageFolder(root=train_path, transform=tf)
    if get == 'test':
        return ImageFolder(root=test_path, transform=tf)
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

if __name__ == '__main__':


    # original resize
    resize_only = transforms.Compose([
        transforms.Resize((Config.RESIZE_SHAPE, Config.RESIZE_SHAPE))
    ])

    # charger les trois datasets avec leur pipelines respectifs
    ds_orig = load_tinygen_image(model=None, tf=resize_only, get='train')
    ds_gn   = load_tinygen_image(model=None, tf=gaussianTF(0, 0.2), get='train')
    ds_comp = load_tinygen_image(model=None, tf=jpegTF(50), get='train')

    
    orig_img, _   = ds_orig[0]    
    gn_tensor, _  = ds_gn[0]     
    cp_tensor, _  = ds_comp[0]   

    def show_pil(img, ax, title):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    def show_tensor(tensor, ax, title):
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img = tensor.numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    # plot
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    show_pil(orig_img, axes[0, 0],'Original')
    show_tensor(gn_tensor, axes[0, 1],'Gaussian Noise')

    show_pil(orig_img,axes[1, 0], 'Original')
    show_tensor(cp_tensor, axes[1, 1],'JPEG Compressed')

    plt.tight_layout()
    plt.show()

        