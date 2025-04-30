import torchvision.transforms as transforms

class Config:
    '''Tous les hyper-paramètres ici'''
    
    DEBUG = False
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
        SAVE_IMAGE = False
        WANDB_LOG = True
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
        if HIDDEN_SIZE_LAB != Dinov2_token_dim[DINOV2_NAME]:
            Adapter = True
        else: Adapter = False
        
    else:
        SAVE_IMAGE = False
        WANDB_LOG = True
        TSNE_LOG = True
        add_tokens_lab = 4
        ADD_TOKENS_LAB = [0, 10, 60, 100, 150]
        ADD_TOKENS_LAB_perf = [0, 10, 30, 50]
        NUM_HIDDEN_LAYER_LLMA_LAB = 6
        HIDDEN_SIZE_LAB = 4096
        BATCH_SIZE_LAB = 16
        LR_LAB = 4e-4
        EPOCHS_LAB = 40
        ITERATION = 1
        DECREASING_LR_LAB = True
        DINOV2_NAME = 'facebook/dinov2-small'
        NHL_LAB = [1, 6, 12, 16]
        
        
        if HIDDEN_SIZE_LAB != Dinov2_token_dim[DINOV2_NAME]:
            Adapter = True
        else: Adapter = False
