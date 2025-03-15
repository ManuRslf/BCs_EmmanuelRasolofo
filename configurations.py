import torchvision.transforms as transforms

'''
    all parameters here
'''

debug = True


if debug:
  ### COMMON
  MODEL = 'midjourney'
  resizeShape = 32
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
  
  
  ### for utils.py & visu
  save_image = False
  wandb_log = False
  ADD_TOKENS_lab = [i for i in range(2, 3)]
  NUM_HIDDEN_LAYER_LLMA_lab = 2
  HIDDEN_SIZE_lab = 768
  BATCH_SIZE_lab = 128
  LR_lab = 1e-3
  EPOCHS_lab = 1

else:

  ### COMMON
  MODEL = 'midjourney'
  resizeShape = 244
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
  
  
  ### for utils.py & visu
  save_image = True
  wandb_log = True
  ADD_TOKENS_lab = [i for i in range(2, 11)]
  NUM_HIDDEN_LAYER_LLMA_lab = 2
  HIDDEN_SIZE_lab = 768
  BATCH_SIZE_lab = 128
  LR_lab = 1e-3
  EPOCHS_lab = 40