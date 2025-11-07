import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def save_output(s_inputs,pred,labels, num,fold,case):

# print(s_inputs.shape)
    
    predict = pred
    predict = F.softmax(predict, dim=1)

    # Get predicted class per pixel
    predict = torch.argmax(predict, dim=1)  # shape: [B, H, W]
  
    # Remove batch dimension if B=1
    predict = predict.squeeze()
    label = labels.squeeze()

    # Convert to numpy for further processing
    
    predict_np = predict.cpu().numpy()
    if case=='test':
     
        s_inputs = s_inputs.cpu().numpy()
        #predict_np= np.argmax(predict, axis=0)
    
        rgb= color_mapping(predict_np)
        #print(rgb.shape)
        rgb= rgb.astype(np.uint8)

        label_rgb= color_mapping(label)

        label_rgb= label_rgb.astype(np.uint8)
        """
        p= predict_np
    

        
        p = torch.from_numpy(p).float()     # convert to torch tensor

        p = p.unsqueeze(0).unsqueeze(0)   # now [1, 1, 360, 250]
        p= F.interpolate(p, size=(1000, 250), mode='nearest')  # resize to original size [1, 1, 1000, 250]
        p = p.squeeze(0).squeeze(0).numpy() # shape [1000, 250]
        """

    
        name= "/home/milkisayebasse/supervised/result/"+str(fold)+ "/" + str(num) + '.png'
        # names= str(num) + "rtoated.png"
        label_names= "/home/milkisayebasse/supervised/result/" +str(fold)+ "/" +str(num) + "_labels.png"
    # dense_names= "D:/important/phd/project/scribble/efficent_u2net/result/scribble/26_aug/" +str(num) + "_dense.png"
        # # # plt.imsave(name,rgb)
        pred_names= "/home/milkisayebasse/supervised/result/" +str(fold)+ "/" +str(num) + "_pred.png"
        plt.imsave(pred_names, rgb)
        plt.imsave(label_names, label_rgb)
        plt.imsave(name,s_inputs.squeeze())


    return predict_np, label
def color_mapping(predict_np):
    green = [0, 255, 0]    # Green
    yellow = [255, 255, 0] # Yellow
    red = [255, 0, 0]      # Red
    blue = [0, 0, 255]     # Blue
    purple = [75,0, 130] # Purple
    orange = [255, 165, 0] # Orange
    # # print(predict_np.shape[0])
    rgb= np.zeros((predict_np.shape[0], predict_np.shape[1], 3), dtype=int)
    rgb[predict_np==0,:]= purple
    rgb[predict_np==3,:]= blue
    rgb[predict_np==2,:]= green
    rgb[predict_np==1,:]= yellow
    rgb[predict_np==4,:]= orange
    rgb[predict_np==5,:]= red
    return rgb;  import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix

def calc_metrics( rs_pred, rs_lab):
    f1_scores = []

  
    label= np.array(rs_lab)
    prediction= np.array(rs_pred)




# Convert the list of true labels to a NumPy array
    label = np.concatenate(label, axis=1)
    true_label_array = np.expand_dims(label, axis=0)

    # Example usage
    num_examples = true_label_array.shape[0]
    height = true_label_array.shape[1]
    width = true_label_array.shape[2]

   #num_class= len(np.unique(true_label_array))
    num_class= int(np.max(true_label_array)) + 1  # including background

    """
    if 0 in np.unique(true_label_array):
        num_class= num_class -1  # ignore background
    """
    prediction= np.concatenate(prediction, axis=1)


    pred_label_array = np.expand_dims(prediction, axis=0)

    avg_recall, avg_precision , avg_accuracy, avg_iou, avg_class_oa  = average_recall_precision(true_label_array, pred_label_array, num_class)
    for i, (r, p) in enumerate(zip(avg_recall, avg_precision)):
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0  # Avoid division by zero
        f1_scores.append(f1)
    average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0  # Avoid division by zero if list is empty
    
    del label, prediction, rs_lab, rs_pred, pred_label_array, true_label_array

    return avg_recall, avg_precision, f1_scores, avg_accuracy, avg_iou, avg_class_oa, average_f1

import numpy as np
from sklearn.metrics import recall_score, precision_score, confusion_matrix

def calculate_recall_precision(y_true, y_pred, num_classes):
    """
    Computes per-class recall, precision, IoU, and class-wise OA
    while ignoring background (class 0).
    Returns arrays of shape (num_classes-1,).
    """

    # 1. Flatten arrays
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()

    # 2. Mask out background (class 0) from both true and predicted labels
    non_zero_indices = (y_true_flat != 0) & (y_pred_flat != 0)
    y_true_non_zero = y_true_flat[non_zero_indices]
    y_pred_non_zero = y_pred_flat[non_zero_indices]

    # 3. Handle case where all pixels are background
    if len(y_true_non_zero) == 0:
        zeros = np.zeros(num_classes - 1, dtype=float)
        return zeros, zeros, 0.0, zeros, zeros

    # 4. Valid (foreground) label range: [1, num_classes-1]
    labels = list(range(1, num_classes))

    # 5. Compute precision and recall per class
    recall = recall_score(y_true_non_zero, y_pred_non_zero,
                          labels=labels, average=None, zero_division=1)
    precision = precision_score(y_true_non_zero, y_pred_non_zero,
                                labels=labels, average=None, zero_division=1)

    # 6. Overall accuracy on non-background pixels
    accuracy = np.mean(y_pred_non_zero == y_true_non_zero)

    # 7. Confusion matrix built only for foreground labels (size (num_classes-1, num_classes-1))
    conf = confusion_matrix(y_true_non_zero, y_pred_non_zero, labels=labels)
    total = conf.sum()

    iou = np.zeros(num_classes - 1, dtype=float)
    class_oa = np.zeros(num_classes - 1, dtype=float)

    # 8. Loop over positions (not label values) to avoid misalignment
    for idx in range(len(labels)):
        TP = conf[idx, idx]
        FP = conf[:, idx].sum() - TP
        FN = conf[idx, :].sum() - TP
        TN = total - (TP + FP + FN)

        denom = TP + FP + FN
        iou[idx] = TP / denom if denom != 0 else 0.0
        class_oa[idx] = (TP + TN) / total if total > 0 else 0.0

    return recall, precision, accuracy, iou, class_oa

def average_recall_precision(y_true_list, y_pred_list, num_classes):
    avg_recall = np.zeros(num_classes-1)
    avg_precision = np.zeros(num_classes-1)
    num_examples = len(y_true_list)
    avg_iou = np.zeros(num_classes-1)
    avg_class_oa = np.zeros(num_classes-1)
    avg_accuracy = 0
   

    for i in range(num_examples):
        recall, precision ,accuracy, iou, class_oa= calculate_recall_precision(y_true_list[i], y_pred_list[i], num_classes)
        avg_recall += recall
        avg_precision += precision
        avg_iou += iou
        avg_class_oa += class_oa
        avg_accuracy += accuracy

    avg_recall /= num_examples
    avg_precision /= num_examples
    avg_iou /= num_examples
    avg_class_oa /= num_examples
    avg_accuracy /= num_examples

    return avg_recall, avg_precision, avg_accuracy, avg_iou, avg_class_oa

"""
def cv_calc(all_fold_recalls,all_fold_precisions,all_fold_accuracies, all_fold_f1, all_fold_ious, all_fold_OAs):
    final_avg_recall = np.mean(all_fold_recalls, axis=0)
    final_avg_precision = np.mean(all_fold_precisions, axis=0)
    final_avg_accuracy = np.mean(all_fold_accuracies)
    final_Av
    std_recall = np.std(all_fold_recalls, axis=0)
    std_precision = np.std(all_fold_precisions, axis=0)
    std_accuracy = np.std(all_fold_accuracies)
    print(std_accuracy,'dtd accuracy')
    f1_scores = []

    print("\n=== Cross-Validation Results ===")
    print(f"Average Accuracy across folds: {final_avg_accuracy * 100:.2f}%")
    
    for i, (r, p, sr, sp) in enumerate(zip(final_avg_recall, final_avg_precision, std_recall, std_precision)):
        print(f"Class {i + 1} - Avg Recall: {r:.4f} ± {sr:.4f}, Avg Precision: {p:.4f} ± {sp:.4f}")
     """   
def cv_calc(all_fold_recalls, all_fold_precisions, all_fold_accuracies, all_fold_f1, all_fold_ious, all_fold_OAs):

        # means
        final_avg_recall    = np.mean(np.asarray(all_fold_recalls),  axis=0)
        final_avg_precision = np.mean(np.asarray(all_fold_precisions),  axis=0)
        final_avg_f1        = np.mean(np.asarray(all_fold_f1), axis=0)
        final_avg_iou       = np.mean(np.asarray(all_fold_ious),axis=0)
        final_avg_oa        = np.mean(np.asarray(all_fold_OAs), axis=0)
        final_avg_accuracy  = np.mean(np.asarray(all_fold_accuracies))

        # stds
        std_recall    = np.std(np.asarray(all_fold_recalls),  axis=0)
        std_precision = np.std(np.asarray(all_fold_precisions),  axis=0)
        std_f1        = np.std(np.asarray(all_fold_f1), axis=0)
        std_iou       = np.std(np.asarray(all_fold_ious),axis=0)
        std_oa        = np.std(np.asarray(all_fold_OAs), axis=0)
        std_accuracy  = np.std(np.asarray(all_fold_accuracies))

        C = final_avg_recall.shape[0]

        print("\n=== Cross-Validation Results ===")
        print(f"Average Accuracy across folds: {final_avg_accuracy * 100:.2f}% (± {std_accuracy*100:.2f}%)\n")
        overall_recall    = np.mean(final_avg_recall)
        overall_precision = np.mean(final_avg_precision)
        overall_f1        = np.mean(final_avg_f1)
        overall_iou       = np.mean(final_avg_iou)
        overall_oa        = np.mean(final_avg_oa)
        for i in range(C):
            print(
                f"Class {i+1} - "
                f"Recall: {final_avg_recall[i]:.4f} ± {std_recall[i]:.4f}, "
                f"Precision: {final_avg_precision[i]:.4f} ± {std_precision[i]:.4f}, "
                f"F1: {final_avg_f1[i]:.4f} ± {std_f1[i]:.4f}, "
                f"IoU: {final_avg_iou[i]:.4f} ± {std_iou[i]:.4f}, "
                f"Class-OA: {final_avg_oa[i]:.4f} ± {std_oa[i]:.4f}"
            )
        print("\n--- Overall (Macro Average across Classes) ---")
        print(f"Recall   : {overall_recall:.4f}")
        print(f"Precision: {overall_precision:.4f}")
        print(f"F1 Score : {overall_f1:.4f}")
        print(f"IoU      : {overall_iou:.4f}")
        print(f"Class-OA : {overall_oa:.4f}")



