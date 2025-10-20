import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_score, recall_score



def calc_metrics( rs_pred, rs_lab):
  
    label= np.array(rs_lab)
    prediction= np.array(rs_pred)




# Convert the list of true labels to a NumPy array
    label = np.concatenate(label, axis=1)
    true_label_array = np.expand_dims(label, axis=0)

    # Example usage
    num_examples = true_label_array.shape[0]
    height = true_label_array.shape[1]
    width = true_label_array.shape[2]
    print(np.unique(true_label_array))
    print(len(np.unique(true_label_array)))
    print(np.unique(prediction))
    num_class= len(np.unique(true_label_array))
    if 0 in np.unique(true_label_array):
        num_class= num_class -1  # ignore background

    prediction= np.concatenate(prediction, axis=1)


    pred_label_array = np.expand_dims(prediction, axis=0)

    avg_recall, avg_precision , avg_accuracy  = average_recall_precision(true_label_array, pred_label_array, num_class)
    
    print(f"Overall Accuracy (ignoring 0 labels): {avg_accuracy* 100:.2f}%")
    for i, (r, p) in enumerate(zip(avg_recall, avg_precision)):
        print(f"Class {i + 1}: Average Recall = {r}, Average Precision = {p}")

    print('||||||||||||||||||||||||||||||||||||||FOLD||||||||||||||||||||||||||||||')
    
    del label, prediction, rs_lab, rs_pred,pred_label_array,true_label_array

    return avg_recall, avg_precision , avg_accuracy

def calculate_recall_precision(y_true, y_pred, num_classes):
    # Flatten the arrays
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()

    # Filter out instances with true label 0
    non_zero_indices = y_true_flat != 0  & (y_pred_flat != 0)
    y_true_non_zero = y_true_flat[non_zero_indices]
    y_pred_non_zero = y_pred_flat[non_zero_indices]

    # Calculate precision and recall for each class
    recall = recall_score(y_true_non_zero, y_pred_non_zero, labels=range(1, num_classes+1), average=None, zero_division=1) # Set zero_division=1 to avoid the warning
    precision = precision_score(y_true_non_zero, y_pred_non_zero, labels=range(1, num_classes+1), average=None,zero_division=1) # Set zero_division=1 to avoid the warning
    accuracy = np.mean(y_pred_non_zero == y_true_non_zero )
    return recall, precision,accuracy

def average_recall_precision(y_true_list, y_pred_list, num_classes):
    avg_recall = np.zeros(num_classes)
    avg_precision = np.zeros(num_classes)
    num_examples = len(y_true_list)
    avg_accuracy=0

    for i in range(num_examples):
        recall, precision ,accuracy= calculate_recall_precision(y_true_list[i], y_pred_list[i], num_classes)
        avg_recall += recall
        avg_precision += precision
        avg_accuracy+=accuracy

    avg_recall /= num_examples
    avg_precision /= num_examples

    return avg_recall, avg_precision,avg_accuracy
def cv_calc(all_fold_recalls,all_fold_precisions,all_fold_accuracies):
    final_avg_recall = np.mean(all_fold_recalls, axis=0)
    final_avg_precision = np.mean(all_fold_precisions, axis=0)
    final_avg_accuracy = np.mean(all_fold_accuracies)
    std_recall = np.std(all_fold_recalls, axis=0)
    std_precision = np.std(all_fold_precisions, axis=0)
    std_accuracy = np.std(all_fold_accuracies)
    print(std_accuracy,'dtd accuracy')
    f1_scores = []

    print("\n=== Cross-Validation Results ===")
    print(f"Average Accuracy across folds: {final_avg_accuracy * 100:.2f}%")
    for i, (r, p) in enumerate(zip(final_avg_recall, final_avg_precision)):
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0  # Avoid division by zero
        f1_scores.append(f1)
        print(f"Class {i + 1} - Avg Recall: {r:.4f}, Avg Precision: {p:.4f}, F1 Score: {f1:.4f}")
    average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0  # Avoid division by zero if list is empty
    print(f"Average F1 Score: {average_f1:.4f}")
    for i, (r, p, sr, sp) in enumerate(zip(final_avg_recall, final_avg_precision, std_recall, std_precision)):
        print(f"Class {i + 1} - Avg Recall: {r:.4f} ± {sr:.4f}, Avg Precision: {p:.4f} ± {sp:.4f}")
    