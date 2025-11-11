from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
seed = 42
np.random.seed(seed)
import torch
from sklearn.model_selection import train_test_split, KFold
def extract_horizontal_patches(data, label, patch_size=64):
  
    import torch
    import numpy as np

    # convert to numpy if torch
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()

    N, H, W = data.shape
    data_patches = []
    label_patches = []

    for i in range(N):
        for x in range(0, W - patch_size + 1, patch_size):
            data_patches.append(data[i, :, x:x + patch_size])
            label_patches.append(label[i, :, x:x + patch_size])

    data_patches = np.array(data_patches)
    label_patches = np.array(label_patches)
    data_patches= np.expand_dims(data_patches, axis=-1)
    label_patches = np.expand_dims(label_patches, axis=-1)
    return data_patches, label_patches

def mc10_data_model():
    import pandas as pd
    data = torch.load('dataset/antarctica_250multi.pt')
    #data = data['test']
    rs_image = data['data'].to('cpu').numpy()
    rs_label = data['label'].to('cpu').numpy()
    
    
   # We’ll generate 3 random folds manually
    folds_1 = []
    
    n_samples = rs_image.shape[0]
    indices = np.arange(n_samples)

    for i in range(3):
        # Randomly shuffle and split into train (70%), val (15%), test (15%)
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=seed + i)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed + i)

        tr_img, va_img, te_img= rs_image[train_idx], rs_image[val_idx], rs_image[test_idx]
        tr_lab, va_lab,  te_lab = rs_label[train_idx], rs_label[val_idx], rs_label[test_idx]
        print(f"Fold {i+1}: Train patches: {tr_img.shape}, Val patches: {va_img.shape}, Test patches: {te_img.shape}")
        folds_1.append({
            'fold': i + 1,
            'train_images': tr_img,  # (M_tr,64,64) or (M_tr,C,64,64) 
            'train_labels': tr_lab,  # (M_tr,64,64)
            'val_images':   va_img,
            'val_labels':   va_lab,
            'test_images':  te_img,
            'test_labels':  te_lab,
        })


    
    folds_2 = [
  
    ]
    
    folds_3 = [

    ]
    fold= [folds_1, folds_2, folds_3]


    model_dir = ['/mnt/data/supervised/eu/merged_sam_lora_fold1_epoch120_average_loss_0.1668_time302627.2_20251110-221445.pth',
        #'/mnt/data/supervised/u2net/greenlandsu2net_u2net_fold2_epoch200_valf1_0.9280_time9071.6_20251030-205930.pth',
       #'/mnt/data/supervised/u2net/greenlandsu2net_u2net_fold3_epoch180_valf1_0.9294_time9074.3_20251030-233045.pth'
                ]
    return fold[0], model_dir


def sharad_manual_data_model():
    import pandas as pd
    data = torch.load('/mnt/data/dataset/sharad_250multi.pt')
    #data = data['test']
    rs_image = data['data'].to('cpu').numpy()
    rs_label = data['label'].to('cpu').numpy()
    """
    folds = [
    (np.arange(0, 80), np.arange(80, 97), np.arange(97, 115)),          # First 151 as test set
    (np.concatenate((np.arange(0, 35), np.arange(70, 115))), np.arange(35, 52), np.arange(52, 70)),  # Middle 151 as test set
    (np.arange(35, 115), np.arange(0, 17), np.arange(17, 35))        # Last 151 as test set
    ]
    """
    folds = [
    (np.arange(0, 80), np.arange(80, 97), np.arange(80, 115)),          # First 151 as test set
    (np.concatenate((np.arange(0, 35), np.arange(70, 115))), np.arange(35, 52), np.arange(35, 70)),  # Middle 151 as test set
    (np.arange(35, 115), np.arange(0, 17), np.arange(0, 35))        # Last 151 as test set
    ]
    patch_size=64
    folds_1 = []
    for fold, (train_idx, val_idx, test_idx) in enumerate(folds):
        #print(f"\nFold {fold + 1}")
        
        # Split images and labels into train/test for the current fold
        train_images, val_images, test_images = rs_image[train_idx], rs_image[val_idx], rs_image[test_idx]
        train_labels, val_labels,  test_labels = rs_label[train_idx], rs_label[val_idx], rs_label[test_idx]
        print(f"Fold {fold+1}: Train patches: {train_images.shape}, Val patches: {val_images.shape}, Test patches: {test_images.shape}")
        
        tr_img, tr_lab = extract_horizontal_patches(train_images, train_labels, patch_size=patch_size)
        va_img, va_lab = extract_horizontal_patches(val_images,   val_labels,   patch_size=patch_size)
        te_img, te_lab = extract_horizontal_patches(test_images,  test_labels,  patch_size=patch_size)
        print(f"Fold {fold+1}: Train patches: {tr_img.shape}, Val patches: {va_img.shape}, Test patches: {te_img.shape}")
        folds_1.append({
            'fold': fold + 1,
            'train_images': tr_img,  # (M_tr,64,64) or (M_tr,C,64,64)
            'train_labels': tr_lab,  # (M_tr,64,64)
            'val_images':   va_img,
            'val_labels':   va_lab,
            'test_images':  te_img,
            'test_labels':  te_lab,
        })


    
    folds_2 = [
  
    ]
    
    folds_3 = [

    ]
    fold= [folds_1, folds_2, folds_3]


    model_dir = ['/mnt/data/supervised/eu/sharad_manualeu_fold1_epoch2140_valf1_0.8335_time16589.2_20251111-000053.pth',
        '/mnt/data/supervised/eu/sharad_manualeu_fold2_epoch2700_valf1_0.9315_time16589.2_20251111-043723.pth',
        '/mnt/data/supervised/eu/sharad_manualeu_fold3_epoch2120_valf1_0.8755_time16588.8_20251111-091351.pth'
                ]
    return fold[0], model_dir

def antarctica_datapatch_model():
    import pandas as pd
    data = torch.load('/mnt/data/dataset/antarctica_250multi.pt')
    #data = data['test']
    rs_image = data['data'].to('cpu').numpy()
    rs_label = data['label'].to('cpu').numpy()
    fold_rs , model_dir = data_rs(rs_image, rs_label)
    return fold_rs , model_dir
def greenland_datapatch_model():
    import pandas as pd
    data = torch.load('/mnt/data/dataset/greenland_250multi.pt')
    #data = data['test']
    rs_image = data['data'].to('cpu').numpy()
    rs_label = data['label'].to('cpu').numpy()
    fold_rs , model_dir = data_rs(rs_image, rs_label)
    return fold_rs , model_dir
def sharad_datapatch_model():
    import pandas as pd
    data = torch.load('/mnt/data/dataset/sharad_250multi.pt')
    #data = data['test']
    rs_image = data['data'].to('cpu').numpy()
    rs_label = data['label'].to('cpu').numpy()
    fold_rs , model_dir = data_rs(rs_image, rs_label)
    return fold_rs , model_dir
def  data_rs(rs_image, rs_label):
  
    
   # We’ll generate 3 random folds manually
    folds_1 = []
    
    n_samples = rs_image.shape[0]
    indices = np.arange(n_samples)
    patch_size=64
    for i in range(3):
        # Randomly shuffle and split into train (70%), val (15%), test (15%)
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=seed + i)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed + i)
        train_images, val_images, test_images = rs_image[train_idx], rs_image[val_idx], rs_image[test_idx]
        train_labels, val_labels,  test_labels = rs_label[train_idx], rs_label[val_idx], rs_label[test_idx]
        tr_img, tr_lab = extract_horizontal_patches(train_images, train_labels, patch_size=patch_size)
        va_img, va_lab = extract_horizontal_patches(val_images,   val_labels,   patch_size=patch_size)
        te_img, te_lab = extract_horizontal_patches(test_images,  test_labels,  patch_size=patch_size)
        print(f"Fold {i+1}: Train patches: {tr_img.shape}, Val patches: {va_img.shape}, Test patches: {te_img.shape}")
        folds_1.append({
            'fold': i + 1,
            'train_images': tr_img,  # (M_tr,64,64) or (M_tr,C,64,64)
            'train_labels': tr_lab,  # (M_tr,64,64)
            'val_images':   va_img,
            'val_labels':   va_lab,
            'test_images':  te_img,
            'test_labels':  te_lab,
        })

    
    folds_2 = [
  
    ]
    
    folds_3 = [

    ]
    fold= [folds_1, folds_2, folds_3]


    model_dir = ['/mnt/data/supervised/sam_lora/merged_sam_lora_fold1_epoch120_average_loss_0.1668_time302627.2_20251110-221445.pth',
        #'/mnt/data/supervised/u2net/greenlandsu2net_u2net_fold2_epoch200_valf1_0.9280_time9071.6_20251030-205930.pth',
       #'/mnt/data/supervised/u2net/greenlandsu2net_u2net_fold3_epoch180_valf1_0.9294_time9074.3_20251030-233045.pth'
    ]
    return fold[0], model_dir
