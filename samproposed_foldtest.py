import os
from skimage import io, transform
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import numpy as np
from PIL import Image
import glob

import cv2
from data_loader import Rescale, RescaleT, RandomCrop,ToTensorLab, SalObjDataset,ResizeInterpolate

from literature.u2net import U2NET # full size version 173.6 MB
from literature.u2net import U2NETP # small version u2net 4.7 MB
from skimage.transform import rotate

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import scipy.io as sio


from implmentation.output import save_output
from implmentation.metrics import calculate_recall_precision
from implmentation.metrics import average_recall_precision
from implmentation.metrics import calc_metrics
from implmentation.metrics import cv_calc
from implmentation.merged import merge_and_resize_folds
from implmentation.dataset import mc10_data_model,antarctica_datapatch_model,greenland_datapatch_model,sharad_datapatch_model
from implmentation.inputs import parse_args, apply_presets, build_model,muti_bce_loss_fusion, PRESETS
seed = 42
np.random.seed(seed)
def  test(test_salobj_dataloader,  model,seg_head, device, fold, case='test',model_name= 'eu'):
    rs_pred = []
    rs_lab = []
    num=0

    model.eval
    for i, data in enumerate(test_salobj_dataloader):
       

                inputs, labels = data['image'], data['label']

                pixel_values= inputs.to(dtype=torch.float32).to(device)
                ground_truth_masks= labels.to(dtype=torch.float32).to(device)

            

                # Forward pass
                image_embeddings = model.image_encoder(pixel_values)
                #resized_ground_truthmask= T.Resize((256, 256))(ground_truth_masks.unsqueeze(1)).float().to(device)
    
                logits_256 = seg_head(image_embeddings)

                predicted_masks = nn.functional.interpolate(
                    logits_256 ,
                    size=(800, 64),
                    mode='bilinear',
                    align_corners=False
                )

                predicted_maskss= predicted_masks.to(dtype=torch.float32)
                probs = torch.softmax(predicted_masks, dim=1)   # convert to class probabilities
                mask = torch.argmax(probs, dim=1)     # [B,H,W] integer mask
                inputs = nn.functional.interpolate(
                    inputs,
                    size=(800, 64),
                    mode='bilinear',
                    align_corners=False
                )
            
 
                rs_pred.append(mask.cpu().numpy().squeeze())
                rs_lab.append(ground_truth_masks.cpu().numpy().squeeze())
    return rs_pred, rs_lab


   

p= 427
def main():
    all_fold_recalls = []
    all_fold_precisions = []
    all_fold_accuracies = []
    all_fold_f1 = []
    all_fold_ious = []
    all_fold_OAs = []   


    folds_a, model_dir = antarctica_datapatch_model()
    folds_g, _ = greenland_datapatch_model()
    folds_s, _ = sharad_datapatch_model()
    merged_folds  = merge_and_resize_folds([folds_a, folds_g, folds_s],
                                        target_h=800, target_w=64,
                                        shuffle=True, seed=42)

    #kf.split(rs_image)

    for fold in merged_folds:
        print(f"\nFold {fold['fold']}")
        
        # Split images and labels into train/test for the current fold
        train_images, val_images, test_images = fold['train_images'], fold['val_images'], fold['test_images']
        train_labels, val_labels,  test_labels = fold['train_labels'], fold['val_labels'], fold['test_labels']




 


        test_salobj_dataset = SalObjDataset(img_name_list = test_images,
                                            lbl_name_list= test_labels,
                                            transform=transforms.Compose([
                                            ToTensorLab(flag=0),                # numpy → torch tensor [C,H,W], likely C=1
                                            ResizeInterpolate(size=(1028, 1028))] )
                                            )
        test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1)

        args = parse_args()
        model_kwargs, criterion = apply_presets(args)
        model, seg_head = build_model(args, model_kwargs)
         # Load the LoRA encoder weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(model_dir[fold['fold'] - 1], map_location=device)

        model.load_state_dict(ckpt["encoder_lora"], strict=False)

        # Load segmentation head weights
        seg_head.load_state_dict(ckpt["seg_head"], strict=True)

        model.to(device).eval()
        seg_head.to(device).eval()


     


        rs_pred=[]
        rs_lab=[] 
     

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.inference_mode():
            rs_pred, rs_lab = test(test_salobj_dataloader, model,seg_head, device, fold, case='test', model_name =args.model)
            
        avg_recall, avg_precision, f1_scores, avg_accuracy, avg_iou, avg_class_oa, average_f1 = calc_metrics(rs_pred, rs_lab)
        print(f"Average F1 Score: {average_f1:.4f}")
        print(f"\nOverall Accuracy (including background): {avg_accuracy * 100:.2f}%")
        for i, (r, p, iou, oa,f1) in enumerate(zip(avg_recall, avg_precision, avg_iou, avg_class_oa,f1_scores)):
            print(f"Class {i+1}: Recall = {r:.4f}, Precision = {p:.4f}, IoU = {iou:.4f}, OA = {oa:.4f}, F1 Score: {f1:.4f}")

        print('||||||||||||||||||||||||||||||||||||||FOLD||||||||||||||||||||||||||||||')

        print(model_dir[fold['fold'] -1])
   
        all_fold_recalls.append(avg_recall)
        all_fold_precisions.append(avg_precision)
        all_fold_accuracies.append(avg_accuracy)
        all_fold_f1.append(f1_scores)
        all_fold_ious.append(avg_iou)
        all_fold_OAs.append(avg_class_oa)
        #||||||||||||||||||||||||||||||||||||||||||||overalll |||||||||||||||||||||||||||||||||||||||||||||||||
    print(np.array(all_fold_recalls).shape,'all fold recall shape')
    print(np.array(all_fold_f1).shape,'all fold recall shape')
    cv_calc(all_fold_recalls,all_fold_precisions,all_fold_accuracies, all_fold_f1, all_fold_ious, all_fold_OAs)
    print("number of test sample is ", rs_image_fold.shape)
    print(args.model, " model")
if __name__ == "__main__":
    main()

