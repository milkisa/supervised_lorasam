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
    data = torch.load('/mnt/data/dataset/antarctica_250multi.pt')
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


    model_dir = ['/mnt/data/supervised/u2net/greenlandsu2net_u2net_fold1_epoch100_valf1_0.9268_time9071.1_20251030-182818.pth',
        '/mnt/data/supervised/u2net/greenlandsu2net_u2net_fold2_epoch200_valf1_0.9280_time9071.6_20251030-205930.pth',
        '/mnt/data/supervised/u2net/greenlandsu2net_u2net_fold3_epoch180_valf1_0.9294_time9074.3_20251030-233045.pth'
                ]
    return fold[0], model_dir



def mc10_datapatch_model():
    import pandas as pd
    data = torch.load('/mnt/data/dataset/antarctica_250multi.pt')
    #data = data['test']
    rs_image = data['data'].to('cpu').numpy()
    rs_label = data['label'].to('cpu').numpy()
    
    
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


    model_dir = ['/mnt/data/supervised/u2net/greenlandsu2net_u2net_fold1_epoch100_valf1_0.9268_time9071.1_20251030-182818.pth',
        '/mnt/data/supervised/u2net/greenlandsu2net_u2net_fold2_epoch200_valf1_0.9280_time9071.6_20251030-205930.pth',
        '/mnt/data/supervised/u2net/greenlandsu2net_u2net_fold3_epoch180_valf1_0.9294_time9074.3_20251030-233045.pth'
                ]
    return fold[0], model_dir

