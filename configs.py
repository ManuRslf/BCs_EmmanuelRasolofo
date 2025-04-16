import torchvision.transforms as transforms

class Config:
    '''Tous les hyper-paramètres ici'''
    
    DEBUG = False
    MODEL = 'midjourney'
    RESIZE_SHAPE = 224

    TRANSFORM = transforms.Compose([
        transforms.Resize((RESIZE_SHAPE, RESIZE_SHAPE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    SHOW_INFO = True
    ADD_TOKENS = 5
    NUM_HIDDEN_LAYER_LLMA = 2
    HIDDEN_SIZE = 768
    BATCH_SIZE = 32
    LR = 0.03
    EPOCHS = 7
    DECREASING_LR = True

    if DEBUG:
        SAVE_IMAGE = False
        WANDB_LOG = False
        ADD_TOKENS_LAB = [i for i in range(2, 3)]
        ADD_TOKENS_LAB_perf = [0, 1]
        NUM_HIDDEN_LAYER_LLMA_LAB = 1
        HIDDEN_SIZE_LAB = 384
        BATCH_SIZE_LAB = 128
        LR_LAB = 1e-3
        EPOCHS_LAB = 1
        ITERATION = 1
        DECREASING_LR_LAB = True
        DINOV2_NAME = 'facebook/dinov2-small'
    else:
        SAVE_IMAGE = False
        WANDB_LOG = True
        add_tokens_lab = 10
        ADD_TOKENS_LAB = [0, 10, 60, 100, 150]
        ADD_TOKENS_LAB_perf = [0, 10, 30, 50]
        NUM_HIDDEN_LAYER_LLMA_LAB = 12
        HIDDEN_SIZE_LAB = 384
        BATCH_SIZE_LAB = 16
        LR_LAB = 4e-4
        EPOCHS_LAB = 30
        ITERATION = 1
        DECREASING_LR_LAB = True
        DINOV2_NAME = 'facebook/dinov2-small'
        NHL_LAB = [1, 6, 12, 15]
