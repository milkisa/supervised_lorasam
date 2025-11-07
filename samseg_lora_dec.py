


# PyTorch and general cache directory
#os.environ["XDG_CACHE_HOME"] = "/home/milkisayebasse/sam/.cache"

def run():


    # ---------------------------
    # Dataset and Processing Setup
    # ---------------------------
    import os
    os.environ['HOME'] = '/home/milkisayebasse/supervised/.cache'
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
    from statistics import mean
    from PIL import Image
    import torchvision.transforms as T
    import monai
    import pandas as pd
    import time
    import torch
    import torch.nn as nn
    import math
    from torchvision import transforms

    from segment_anything import sam_model_registry
    from data_loader import Rescale, RescaleT, RandomCrop,ToTensorLab, SalObjDataset,ResizeInterpolate
    from sam_lora.lora_tune import LoRALinear
    from sam_lora.decoder_head import ASPPHead,ASPPHeadStrong
    
    

    k = 0
    data = torch.load('/mnt/data/dataset/mc_250_fold_3.pt')
    #data = torch.load('/mnt/data/dataset/greenland_250_bestfold_2.pt')
    data = torch.load('/mnt/data/dataset/greenland_64multi.pt')
    #data = data['test']
    rs_bed = data['data'].to('cpu').numpy()[:1470,]
    rs_bed_gt = data['label'].to('cpu').numpy()[:1470,]

    rs_image_fold = rs_bed
    rs_label_fold = rs_bed_gt
    batch_size_train = 1  # your setting

    salobj_dataset = SalObjDataset(
        img_name_list=rs_image_fold,
        lbl_name_list=rs_label_fold,
        transform=transforms.Compose([
            ToTensorLab(flag=0),                # numpy → torch tensor [C,H,W], likely C=1
            ResizeInterpolate(size=(1028, 1028))  # resize image & label consistently
        ])
    )

    salobj_dataloader = DataLoader(
        salobj_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=1
    )


    # ---------------------------
    checkpoint_path = "/mnt/data/checkpoint/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    model = sam
   
    def inject_lora_into_sam_image_encoder(image_encoder, r=8, alpha=16, dropout=0.1):
    
                

        for block in image_encoder.blocks:
            attn = block.attn
            attn.qkv = LoRALinear(attn.qkv, r=r, alpha=alpha, dropout=dropout)

        return image_encoder
    # Apply LoRA to the image_encoder

# Inject LoRA into the image encoder
    model.image_encoder = inject_lora_into_sam_image_encoder(
        sam.image_encoder, r=8, alpha=16, dropout=0.1
    )

    # Lightweight, promptless segmentation head
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_class=5
    seg_head = ASPPHead(in_ch=256, out_ch=num_class).to(device)
    #seg_head = ASPPHeadStrong(in_ch=256, out_ch=num_class, rates=(1,6,12,18), low_ch=64, return_aux=True).to(device)
    # --- Freeze everything except LoRA params in the encoder ---
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

    seg_loss = monai.losses.DiceCELoss(
        softmax=True,          # <-- NOT sigmoid
        to_onehot_y=True,      # convert gt indices to one-hot internally
        include_background=True,
        squared_pred=True,
        reduction='mean'
                )

    # ---------------------------
    # Training Loop
    # ---------------------------
    num_epochs = 1000
    def count_parameters(m, trainable_only=False):
        return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only))

    # Encoder (with LoRA)
    total_encoder_params = count_parameters(model, trainable_only=False)
    trainable_encoder_params = count_parameters(model, trainable_only=True)

    # Segmentation head
    total_head_params = count_parameters(seg_head, trainable_only=False)
    trainable_head_params = count_parameters(seg_head, trainable_only=True)

    # Totals
    total_params = total_encoder_params + total_head_params
    trainable_params = trainable_encoder_params + trainable_head_params

    print(f"  🧮 Total parameters       : {total_params:,}")
    print(f"  🏋️ Trainable parameters   : {trainable_params:,}")



    model.to(device)
    start_time = time.time()

    model_dir = '/mnt/data/supervised/sam_lora_dec/'
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(num_epochs+1):
        model.train()
        epoch_losses = []

        for i, data in enumerate(salobj_dataloader):
            inputs, labels = data['image'], data['label']
            pixel_values= inputs.to(device, dtype=torch.float32)
            ground_truth_masks= labels.to(device, dtype=torch.float32)

           

            # Forward pass
            image_embeddings = model.image_encoder(pixel_values)
            #resized_ground_truthmask= T.Resize((256, 256))(ground_truth_masks.unsqueeze(1)).float().to(device)
  
            logits_256 = seg_head(image_embeddings)
            #logits_256,_ = seg_head(image_embeddings)  # [B, 1 or C, H_in, W_in]


            predicted_masks = nn.functional.interpolate(
                logits_256 ,
                size=(1200, 64),
                mode='bilinear',
                align_corners=False
            )
            predicted_maskss= predicted_masks.to(dtype=torch.float32)
          #  print(predicted_maskss.shape, ground_truth_masks.shape, 'shapes for loss')
            loss = seg_loss(predicted_masks, ground_truth_masks)
            epoch_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0 :
            ckpt_path = os.path.join(
                model_dir, f" greenland_testmultiv_64_{epoch+1}_t{time.time()-start_time:.1f}s.pth"
            )
            torch.save({
                "epoch": epoch + 1,
                "encoder_lora": model.state_dict(),     # SAM (with LoRA-injected layers)
                "seg_head": seg_head.state_dict(),      # your lightweight decoder
                "optimizer": optimizer.state_dict(),
            }, ckpt_path)



        print(f'EPOCH: {epoch} | Mean loss: {mean(epoch_losses):.4f}')


if __name__ == '__main__':
    print("Start Training")
    run()