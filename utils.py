import os 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import configurations

#Return : dataset of the chosen file directory
def TinygenImage(model:str=None, tf:transforms.Compose=None):
     
    '''model : none -> full dataset'''
    MODEL_NAME = ['biggan', 'vqdm', 'sdv5', 'wukong', 'adm', 'glide', 'midjourney']
    
    path_to_data = [r'DATA\tinygenimage\imagenet_ai_0419_biggan', 
                    r'DATA\tinygenimage\imagenet_ai_0419_vqdm', 
                    r'DATA\tinygenimage\imagenet_ai_0424_sdv5',
                    r'DATA\tinygenimage\imagenet_ai_0424_wukong',
                    r'DATA\tinygenimage\imagenet_ai_0508_adm',
                    r'DATA\tinygenimage\imagenet_glide',
                    r'DATA\tinygenimage\imagenet_midjourney'
                    ]
    
    
    valid_model = {
        MODEL_NAME[i] : path_to_data[i] for i in range(7)
    }
    
    if model not in MODEL_NAME and model is not None: 
        raise ValueError(f'Model not valid. {valid_model}')
        
    #get actual path
    base_path = os.getcwd()
    
    if model is None:
        
        #use merged data
        merged_path = r'DATA\tinygenimage_merged'
        path_final_train = os.path.join(base_path, merged_path, 'train').replace("\\", "/")
        path_final_test = os.path.join(base_path, merged_path, 'test').replace("\\", "/")
        return ImageFolder(root=path_final_train, transform=tf), ImageFolder(root=path_final_test, transform=tf)
        
    
    else:
        
        path_final_train = os.path.join(base_path, valid_model[model], 'train').replace("\\", "/")
        path_final_test = os.path.join(base_path, valid_model[model], 'test').replace("\\", "/")

        return ImageFolder(root=path_final_train, transform=tf), ImageFolder(root=path_final_test, transform=tf)
    
def verbose(show:bool=True):
    if show:
        print(f"Images are resized {configurations.resizeShape}x{configurations.resizeShape}")
        print(f"Additional token added : {configurations.ADD_TOKENS}")
        print(f"Hidden layer LLMA : {configurations.NUM_HIDDEN_LAYER_LLMA} with size {configurations.HIDDEN_SIZE}")
        print(f"Batch size : {configurations.BATCH_SIZE}, LR : {configurations.LR}, epochs : {configurations.EPOCHS}")
    



if __name__ == '__main__':
    
    tf = transforms.Compose(
        [
            transforms.Resize((1024,1024)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ]
    )
    
    TinygenImage(tf=tf)