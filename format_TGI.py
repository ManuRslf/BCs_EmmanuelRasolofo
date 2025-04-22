import kagglehub
import os
import shutil
#download
print("Telechargement du dataset.")

source = kagglehub.dataset_download("yangsangtai/tiny-genimage")


destination = os.path.join(os.getcwd(), r'DATA\tinygenimage')

os.makedirs(destination, exist_ok=True)

print(f"Path source : {source}")
print(f"Destination : {destination}")

#mv
for filename in os.listdir(source):
    src_file = os.path.join(source, filename)
    
    dst_file = os.path.join(destination, filename)
    shutil.move(src_file, dst_file)




print("Déplacement terminé.")

TRAIN = 'train'
TEST = 'test'
VAL = 'val'

#creation du dataset merged
print("Creation de tinygenimage_merged.")
destination_merged = os.path.join(os.getcwd(), r'DATA\tinygenimage_merged')
destination_mergedTRAIN_ai = os.path.join(os.getcwd(), r'DATA\tinygenimage_merged\train\ai')
destination_mergedTRAIN_nature = os.path.join(os.getcwd(), r'DATA\tinygenimage_merged\train\nature')

destination_mergedTEST_ai = os.path.join(os.getcwd(), r'DATA\tinygenimage_merged\test\ai')
destination_mergedTEST_nature = os.path.join(os.getcwd(), r'DATA\tinygenimage_merged\test\nature')


os.makedirs(destination_merged, exist_ok=True)
os.makedirs(destination_mergedTRAIN_ai, exist_ok=True)
os.makedirs(destination_mergedTRAIN_nature, exist_ok=True)
os.makedirs(destination_mergedTEST_ai, exist_ok=True)
os.makedirs(destination_mergedTEST_nature, exist_ok=True)



dst_test_ai = os.path.join(os.getcwd(), destination_mergedTEST_ai)
dst_test_nature = os.path.join(os.getcwd(), destination_mergedTEST_nature)

dst_train_ai = os.path.join(os.getcwd(), destination_mergedTRAIN_ai)
dst_train_nature = os.path.join(os.getcwd(), destination_mergedTRAIN_nature)


# Parcours de chaque model dans tinygenimage
for filename in os.listdir(destination):
    model_path = os.path.join(destination, filename)
    
    # parcours de train et val dans chaque model
    for tr_val in os.listdir(model_path):
        if tr_val == VAL:
            
            vl_path = os.path.join(model_path, tr_val)
            
            #parcours de ai et nature
            for ai_nature in os.listdir(vl_path):
                src_file2 = os.path.join(vl_path, ai_nature)
                print(f"Copie de {src_file2} dans tinygenimage_merged")
                if ai_nature == 'ai':
                    shutil.copytree(src_file2, dst_test_ai, dirs_exist_ok=True)
                
                    
                else:
                    shutil.copytree(src_file2, dst_test_nature, dirs_exist_ok=True)
            
            #et on rennome le dossier de val en train
            os.rename(os.path.join(model_path, tr_val), os.path.join(model_path, TEST))
            
        if tr_val == TRAIN:
            tr_path = os.path.join(model_path, tr_val)
            
            #parcours de ai et nature
            for ai_nature in os.listdir(tr_path):
                src_file2 = os.path.join(tr_path, ai_nature)
                print(f"Copie de {src_file2} dans tinygenimage_merged")
                if ai_nature == 'ai':
                    shutil.copytree(src_file2, dst_train_ai, dirs_exist_ok=True)
                
                    
                else:
                    shutil.copytree(src_file2, dst_train_nature, dirs_exist_ok=True)
            
    
print("Creation de merged terminé.")

