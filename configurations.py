import torchvision.transforms as transforms

'''
    all parameters here
'''


### COMMON
MODEL = 'midjourney'
resizeShape = 224
tf = transforms.Compose(
    [
        transforms.Resize((resizeShape,resizeShape)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ]
)


#### for custom_model.py
show_info = True
ADD_TOKENS = 5
NUM_HIDDEN_LAYER_LLMA = 2
HIDDEN_SIZE = 768
BATCH_SIZE = 32
LR = 0.03
EPOCHS = 7


### for lab.py
save_image = True
ADD_TOKENS_lab =  [i for i in range(2, 4)]
NUM_HIDDEN_LAYER_LLMA_lab = 2
HIDDEN_SIZE_lab = 768
BATCH_SIZE_lab = 32
LR_lab = 0.03
EPOCHS_lab = 2