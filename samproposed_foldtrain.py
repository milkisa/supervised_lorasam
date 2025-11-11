
import os
os.environ['HOME'] = '/home/milkisayebasse/sparse/.cache'
print('yaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa||||||||||||||||||||||||||||||||||||')
import torch
import torchvision
import copy
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from implmentation.merged import merge_and_resize_folds
import matplotlib.pyplot as plt
from data_loader import ToTensorLab,SalObjDataset
from literature.aspp import UNetASPP
import monai
from implmentation.dataset import mc10_data_model,antarctica_datapatch_model,greenland_datapatch_model,sharad_datapatch_model,sharad_manual_data_model
from implmentation.inputs import parse_args, apply_presets, build_model,muti_bce_loss_fusion, PRESETS
from samproposed_foldtest import test
from implmentation.metrics import calc_metrics
from implmentation.metrics import calc_metrics
from data_loader import Rescale, RescaleT, RandomCrop,ToTensorLab, SalObjDataset,ResizeInterpolate
import time
import scipy.io as sio

import torchvision.transforms as T
from sklearn.model_selection import KFold
seed = 42
np.random.seed(seed)
#folds, _ = antarctica_datapatch_model()
folds_a, _ = antarctica_datapatch_model()
folds_g, _ = greenland_datapatch_model()
folds_s, _ = sharad_manual_data_model()
merged_folds = merge_and_resize_folds([folds_a, folds_g, folds_s],
                                      target_h=800, target_w=64,
                                      shuffle=True, seed=42)

print("Total merged folds:", len(merged_folds))

#kf.split(rs_image)
for fold in merged_folds:
    print(f"\nFold {fold['fold']}")
    
    # Split images and labels into train/test for the current fold
    train_images, val_images, test_images = fold['train_images'], fold['val_images'], fold['test_images']
    train_labels, val_labels,  test_labels = fold['train_labels'], fold['val_labels'], fold['test_labels']
    
    # Display the shapes of the training and testing data
    print("Train Images shape:", train_images.shape)
    print("Train val shape:", val_images.shape)
    print("test_images shape:", test_images.shape)



    start_time= time.time()
    args = parse_args()
    model_kwargs, criterion = apply_presets(args)
    model, seg_head = build_model(args, model_kwargs)
    preset = PRESETS[args.model]
    save_dir = preset.get("model_dir") 
    # make sure the folder exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # Only add betas/eps if they exist
    if getattr(args, "betas", None) is not None:
        opt_kwargs["betas"] = tuple(args.betas)
    if getattr(args, "eps", None) is not None:
        opt_kwargs["eps"] = args.eps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    seg_head.to(device)
    for name, p in model.named_parameters():
        p.requires_grad = ("lora" in name)

    # Ensure seg head is trainable
    for p in seg_head.parameters():
        p.requires_grad = True

    # --- Optimizer & loss ---
    # Parameter groups: smaller LR for LoRA, larger for the head (learns faster)
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n]
    head_params = list(seg_head.parameters())
    
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": 5e-5, "weight_decay": 0.01},
            {"params": head_params, "lr": 1e-4, "weight_decay": 1e-4},
        ]
    )

     
    # (Optional) AMP scaler for stability/speed on GPU
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # --- (Optional) param counts for sanity ---
    def count_params(m, trainable_only=False):
        return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only))

    print(f"Trainable (LoRA encoder): {sum(p.numel() for p in lora_params):,}")
    print(f"Trainable (seg head):     {sum(p.numel() for p in head_params):,}")
    print(f"Total trainable:          {count_params(model, True) + sum(p.numel() for p in seg_head.parameters()):,}")
       # Encoder (with LoRA)
    total_encoder_params = count_params(model, trainable_only=False)
    trainable_encoder_params = count_params(model, trainable_only=True)

    # Segmentation head
    total_head_params = count_params(seg_head, trainable_only=False)
    trainable_head_params = count_params(seg_head, trainable_only=True)

    # Totals
    total_params = total_encoder_params + total_head_params
    trainable_params = trainable_encoder_params + trainable_head_params

    print(f"  🧮 Total parameters       : {total_params:,}")
    print(f"  🏋️ Trainable parameters   : {trainable_params:,}")
    seg_loss = monai.losses.DiceCELoss(
        softmax=True,          # <-- NOT sigmoid
        to_onehot_y=True,      # convert gt indices to one-hot internally
        include_background=True,
        squared_pred=True,
        reduction='mean'
                )

  
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    epoch_num = args.epochs
    batch_size_train = args.batch_size_train
    salobj_dataset = SalObjDataset(
        img_name_list=train_images,
        lbl_name_list= train_labels,
        transform=transforms.Compose([
            ToTensorLab(flag=0),                # numpy → torch tensor [C,H,W], likely C=1
            ResizeInterpolate(size=(1028, 1028))] )) # resize image & label consistently
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)
    val_salobj_dataset = SalObjDataset(img_name_list = val_images,
                                            lbl_name_list= val_labels,
                                        transform=transforms.Compose([
                                            ToTensorLab(flag=0),                # numpy → torch tensor [C,H,W], likely C=1
                                            ResizeInterpolate(size=(1028, 1028))] ))
    val_salobj_dataloader = DataLoader(val_salobj_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1)
            
    
    best_val_f1 = 0.0
    best_state_dict = None  # will hold a deepcopy of the best weights
   
    for epoch in range(0, epoch_num+1):
        model.train()
        seg_head.train()
        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']
            inputs = inputs.type(torch.FloatTensor)
            #inputs = inputs.repeat(1, 3, 1, 1)
            labels = labels.type(torch.FloatTensor)
            inputs_v, labels_v = inputs.to(device), labels.to(device)
           # Forward pass
            image_embeddings = model.image_encoder(inputs_v)
            #resized_ground_truthmask= T.Resize((256, 256))(ground_truth_masks.unsqueeze(1)).float().to(device)
  
            logits_256 = seg_head(image_embeddings)
            #logits_256,_ = seg_head(image_embeddings)  # [B, 1 or C, H_in, W_in]


            predicted_masks = nn.functional.interpolate(
                logits_256 ,
                size=(800, 64),
                mode='bilinear',
                align_corners=False
            )
            predicted_maskss= predicted_masks.to(dtype=torch.float32)
          #  print(predicted_maskss.shape, ground_truth_masks.shape, 'shapes for loss')
            loss = seg_loss(predicted_masks, labels_v)
          

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()




            # # print statistics
            running_loss += loss.data.item()

            # del temporary outputs and loss
            del  loss

            

        if epoch % 40 == 0:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            avg_loss = running_loss / max(1, ite_num4val)
    
            print(f"[Fold {fold['fold']}][Epoch {epoch:03d}/{epoch_num:03d}] "
                  f"Train Loss: {avg_loss:.4f} "
                  ) 
            ckpt_name = (
                f"mergedmanuals_{args.model}_fold{fold['fold']}_epoch{epoch}"
                f"_average_loss_{avg_loss:.4f}_time{time.time()-start_time:.1f}_{timestamp}.pth"
            )
            ckpt_path = os.path.join(save_dir, ckpt_name)
            torch.save({
                "encoder_lora": model.state_dict(),     # SAM (with LoRA-injected layers)
                "seg_head": seg_head.state_dict(),      # your lightweight decoder
            }, ckpt_path)
            print(f"  >> ✅ Saved checkpoint to: {ckpt_path}")
            """
            with torch.inference_mode(): 
                print('==================== inference mode ====================')
                rspred, rs_lab= test(val_salobj_dataloader, model,seg_head, device, fold['fold'], case='val', model_name =args.model)
                avg_recall, avg_precision, f1_scores, avg_accuracy, avg_iou, avg_class_oa, average_f1 = calc_metrics(rspred, rs_lab)
                # ----- Build safe save path -----
                
                if average_f1 > best_val_f1:
                    print(f"  >> New best model found at epoch {epoch:03d} (val f1: {average_f1:.4f} > {best_val_f1:.4f}), saving...")
                    best_val_f1 = average_f1
                    best_epoch = epoch
                    best_avg_accuracy = avg_accuracy
                    #best_enc_dict = copy.deepcopy(model.state_dict())  # cache weights (no disk I/O)
                    #best_dec_dict = copy.deepcopy(seg_head.state_dict())  # cache weights (no disk I/O)
                    # ⚠️ move tensors to CPU to avoid doubling GPU memory
                    best_enc_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    best_dec_dict = {k: v.detach().cpu() for k, v in seg_head.state_dict().items()}

                    # free any temporary CUDA buffers
                    torch.cuda.empty_cache()
                """         
                


        # reset trackers
        running_loss = 0.0
        running_tar_loss = 0.0
        ite_num4val = 0
    """
    print(f"=== Fold {fold['fold']} training complete in {(time.time()-start_time)/60:.2f} mins ===")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ckpt_name = (
        f"greenland_{args.model}_fold{fold['fold']}_epoch{best_epoch}"
        f"_valf1_{best_val_f1:.4f}_time{time.time()-start_time:.1f}_{timestamp}.pth"
    )

    print(f"  >> Val @ epoch {best_epoch:03d}: acc={best_avg_accuracy:.4f}, f1={best_val_f1:.4f}")


    ckpt_path = os.path.join(save_dir, ckpt_name)
    torch.save(best_state_dict, ckpt_path)
    torch.save({
                "encoder_lora": best_enc_dict,     # SAM (with LoRA-injected layers)
                "seg_head": best_dec_dict,      # your lightweight decoder
            }, ckpt_path)

    print(f"  >> ✅ Saved (best val f1 so far: {best_val_f1:.4f}) to: {ckpt_path}")

    """
        


        

       
       
        


