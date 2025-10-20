import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def save_output(s_inputs,pred,labels, num,fold):


   
    rgb= color_mapping(pred)
    #print(rgb.shape)
    rgb= rgb.astype(np.uint8)

    label_rgb= color_mapping(labels)

    label_rgb= label_rgb.astype(np.uint8)

    
    name= "/home/milkisayebasse/supervised/result/"+str(fold)+ "/" + str(num) + '.png'
    # names= str(num) + "rtoated.png"
    label_names= "/home/milkisayebasse/supervised/result/" +str(fold)+ "/" +str(num) + "_labels.png"
# dense_names= "D:/important/phd/project/scribble/efficent_u2net/result/scribble/26_aug/" +str(num) + "_dense.png"
    # # # plt.imsave(name,rgb)
    pred_names= "/home/milkisayebasse/supervised/result/" +str(fold)+ "/" +str(num) + "_pred.png"

    plt.imsave(pred_names, rgb)
    plt.imsave(label_names, label_rgb)
    plt.imsave(name, np.transpose(s_inputs, (1,2,0)))
    



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
    rgb[predict_np==3,:]= yellow
    rgb[predict_np==2,:]= green
    rgb[predict_np==1,:]= blue
    rgb[predict_np==4,:]= red
    rgb[predict_np==5,:]= red
    return rgb;  