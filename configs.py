import torchvision.transforms as transforms
import torch
from PIL import Image
import io

class Config:
    '''Tous les hyper-paramètres ici'''
    
    DEBUG = False
    Adapter_EXTERN = False
    
    MODEL = 'midjourney'
    RESIZE_SHAPE = 224
    Dinov2_token_dim = {
        'facebook/dinov2-base' : 768,
        'facebook/dinov2-small' : 384
    }
    TRANSFORM = transforms.Compose([
        transforms.Resize((RESIZE_SHAPE, RESIZE_SHAPE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    SHOW_INFO = True

    if DEBUG:
        RESIZE_SHAPE = 28
        SAVE_IMAGE = False
        WANDB_LOG = False
        TSNE_LOG = False
        add_tokens_lab = 1
        ADD_TOKENS_LAB = [i for i in range(2, 3)]
        ADD_TOKENS_LAB_perf = [0, 1]
        NUM_HIDDEN_LAYER_LLMA_LAB = 1
        HIDDEN_SIZE_LAB = 384
        BATCH_SIZE_LAB = 8
        LR_LAB = 1e-3
        EPOCHS_LAB = 1
        ITERATION = 1
        DECREASING_LR_LAB = True
        DINOV2_NAME = 'facebook/dinov2-small'
        NHL_LAB = [1, 6]
        HSL_LAB = [128, 256]
        EPOCHS_HSL = [1, 2]
        STD_GAUSSIAN_NOISE = [0.01, 1]
        QUALITY_JPEG_COMPRESSION = [100, 95]
        if HIDDEN_SIZE_LAB != Dinov2_token_dim[DINOV2_NAME] and not Adapter_EXTERN:
            Adapter = True
        elif HIDDEN_SIZE_LAB == Dinov2_token_dim[DINOV2_NAME] and not Adapter_EXTERN: Adapter = False
        
    else:
        SAVE_IMAGE = False
        WANDB_LOG = True
        TSNE_LOG = False
        add_tokens_lab = 30
        ADD_TOKENS_LAB = [0, 10, 60, 100, 150]
        ADD_TOKENS_LAB_perf = [0, 10, 30, 50]
        NUM_HIDDEN_LAYER_LLMA_LAB = 2
        HIDDEN_SIZE_LAB = 768
        BATCH_SIZE_LAB = 16
        LR_LAB = 4e-4
        # epoch plus grand si taille llama different de dinov2
        EPOCHS_LAB = 120
        ITERATION =2
        DECREASING_LR_LAB = True
        DINOV2_NAME = 'facebook/dinov2-base'
        NHL_LAB = [1, 6, 12, 16]
        HSL_LAB = [384, 768, 1536]
        EPOCHS_HSL = [100, 200, 300]
        STD_GAUSSIAN_NOISE = [0.01, 0.05, 0.1, 0.3, 0.5, 1]
        QUALITY_JPEG_COMPRESSION = [100, 95, 85, 70, 50, 30, 10, 1]
        
        if HIDDEN_SIZE_LAB != Dinov2_token_dim[DINOV2_NAME] and not Adapter_EXTERN:
            Adapter = True
        elif HIDDEN_SIZE_LAB == Dinov2_token_dim[DINOV2_NAME] and not Adapter_EXTERN: Adapter = False
        



# methode custom pour la pipeline gaussian nois
class GaussianNoise:
    def __init__(self, mean:float=0, std:float=1):
        self.mean = mean
        self.std = std
        
    # on va rendre l'instance de l'objet en callable
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean
    
    
# methode custom pour compresser l'image en jpeg
class JPEGcompression:
    def __init__(self, quality:int=50):
        self.quality=quality
        
    def __call__(self, img):
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=self.quality)
        buf.seek(0)
        return Image.open(buf)

# Deux methodes get qui va retourner la pipeline correspondant de la transformation voulue
def gaussianTF(mean:float=0, std:float=1):
    TRANSFORM_GN = transforms.Compose([
        transforms.Resize((Config.RESIZE_SHAPE, Config.RESIZE_SHAPE)),
        transforms.ToTensor(),
        GaussianNoise(mean=mean, std=std),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    return TRANSFORM_GN


def jpegTF(quality:int=50):
    TRANSFORM_COMPRESSED = transforms.Compose([
        transforms.Resize((Config.RESIZE_SHAPE, Config.RESIZE_SHAPE)),
        JPEGcompression(quality=quality),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    return TRANSFORM_COMPRESSED