import torchvision.transforms as transforms

'''
    all parameters here
'''


MODEL = 'biggan'
tf = transforms.Compose(
    [
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ]
)
NUM_HIDDEN_LAYER_LLMA = 2
HIDDEN_SIZE = 768

BATCH_SIZE = 32
LR = 0.02
EPOCHS = 5