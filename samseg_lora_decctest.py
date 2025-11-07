


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
    from sam_codes.metrics import calc_metrics,calculate_recall_precision,cv_calc
    from sam_codes.output import save_output
    import math
    from torchvision import transforms

    from segment_anything import sam_model_registry
    from data_loader import Rescale, RescaleT, RandomCrop,ToTensorLab, SalObjDataset,ResizeInterpolate
    from sam_lora.lora_tune import LoRALinear
    from sam_lora.decoder_head import ASPPHead
    
    

    k = 0
   # 
    #data = torch.load('/mnt/data/dataset/greenland_250_bestfold_2.pt')
   
    data = torch.load('/mnt/data/dataset/greenland_64multi.pt')
    #data = data['test']
    rs_bed = data['data'].to('cpu').numpy()[1470:,]
    rs_bed_gt = data['label'].to('cpu').numpy()[1470:,]
    print(rs_bed.shape, rs_bed_gt.shape, 'data shapes|||||||||||||||||||||||||||||||||||||||||||||')
    print(np.unique(rs_bed_gt), 'unique labels|||||||||||||||||||||||||||||||||||||||||||||')
    """
    data = torch.load('/mnt/data/dataset/mc_250_fold_1.pt')
    data= data['test']
    print(data.keys())
    rs_bed = data['bedrock'].to('cpu').numpy()
    rs_bed_gt = data['bed_gt'].to('cpu').numpy()
    """

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
        shuffle=False,
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
    ckpt = torch.load("/mnt/data/supervised/sam_lora_dec/ greenland_testmultiv_64_21_t22795.8s.pth", map_location=device)

    # Load the LoRA encoder weights
    sam.load_state_dict(ckpt["encoder_lora"], strict=False)

    # Load segmentation head weights
    seg_head.load_state_dict(ckpt["seg_head"], strict=True)

    sam.to(device).eval()
    seg_head.eval()

    print("✅ LoRA encoder and segmentation head successfully loaded.")


    
   
    model.to(device)
    start_time = time.time()

    model_dir = '/mnt/data/supervised/sam_lora_dec/'
    os.makedirs(model_dir, exist_ok=True)
    rs_pred= []
    rs_lab= []
    all_fold_recalls = []
    all_fold_precisions = []
    all_fold_accuracies = []
    with torch.no_grad():
     
            model.eval()
           

            for i, data in enumerate(salobj_dataloader):
       

                inputs, labels = data['image'], data['label']
                pixel_values= inputs.to(device, dtype=torch.float32)
                ground_truth_masks= labels.to(device, dtype=torch.float32)

            

                # Forward pass
                image_embeddings = model.image_encoder(pixel_values)
                #resized_ground_truthmask= T.Resize((256, 256))(ground_truth_masks.unsqueeze(1)).float().to(device)
    
                logits_256 = seg_head(image_embeddings)

                predicted_masks = nn.functional.interpolate(
                    logits_256 ,
                    size=(1200, 64),
                    mode='bilinear',
                    align_corners=False
                )

                predicted_maskss= predicted_masks.to(dtype=torch.float32)
                probs = torch.softmax(predicted_masks, dim=1)   # convert to class probabilities
                mask = torch.argmax(probs, dim=1)     # [B,H,W] integer mask
                inputs = nn.functional.interpolate(
                    inputs,
                    size=(1200, 64),
                    mode='bilinear',
                    align_corners=False
                )
                save_output(inputs.cpu().numpy().squeeze(), mask.cpu().numpy().squeeze(), ground_truth_masks.cpu().numpy().squeeze(), i, fold=4)
            
 
                rs_pred.append(mask.cpu().numpy().squeeze())
                rs_lab.append(ground_truth_masks.cpu().numpy().squeeze())
                
    print(np.array(rs_pred).shape, np.array(rs_lab).shape, 'final shapes|||||||||||||||||||||||||||||||||||||||||')
    avg_recall, avg_precision, f1_scores, avg_accuracy, avg_iou, avg_class_oa, average_f1 = calc_metrics(rs_pred, rs_lab)
    print(f"Average F1 Score: {average_f1:.4f}")
    print(f"\nOverall Accuracy (including background): {avg_accuracy * 100:.2f}%")
    for i, (r, p, iou, oa,f1) in enumerate(zip(avg_recall, avg_precision, avg_iou, avg_class_oa,f1_scores)):
        print(f"Class {i+1}: Recall = {r:.4f}, Precision = {p:.4f}, IoU = {iou:.4f}, OA = {oa:.4f}, F1 Score: {f1:.4f}")

    print('||||||||||||||||||||||||||||||||||||||FOLD||||||||||||||||||||||||||||||')
   

if __name__ == '__main__':
    print("Start TEsting")
    run()