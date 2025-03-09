import torchvision.transforms as transforms

'''
    all parameters here
'''


MODEL = 'biggan'
tf = transforms.Compose(
    [
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ]
)


ADD_TOKENS = 3
NUM_HIDDEN_LAYER_LLMA = 2
HIDDEN_SIZE = 32

BATCH_SIZE = 8
LR = 0.02
EPOCHS = 2