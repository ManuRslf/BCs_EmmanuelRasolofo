import torchvision.transforms as transforms

'''
    all parameters here
'''

show_info = True
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


ADD_TOKENS = 5
NUM_HIDDEN_LAYER_LLMA = 2
HIDDEN_SIZE = 768

BATCH_SIZE = 32
LR = 0.03
EPOCHS = 7