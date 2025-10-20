# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

#==========================dataset load==========================
class RescaleT(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# img = transform.resize(image,(new_h,new_w),mode='constant')
		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
		lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

		return {'imidx':imidx, 'image':img,'label':lbl}

class Rescale(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']

		if random.random() >= 0.5:
			image = image[::-1]
			label = label[::-1]

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		img = transform.resize(image,(new_h,new_w),mode='constant')
		lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		return {'imidx':imidx, 'image':img,'label':lbl}

class RandomCrop(object):

	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']

		if random.random() >= 0.5:
			image = image[::-1]
			label = label[::-1]

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		image = image[top: top + new_h, left: left + new_w]
		label = label[top: top + new_h, left: left + new_w]

		return {'imidx':imidx,'image':image, 'label':label}



class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

		imidx, image, label =sample['imidx'], sample['image'], sample['label']

		tmpLbl = np.zeros(label.shape)

		# if(np.max(label)<1e-6):
		# 	label = label
		# else:
		# 	label = label/np.max(label)

		# change the color space
		# print("absolute", np.max(np.abs(image)))
		# print("real", np.max(image))
		image = image/np.max(np.abs(image))


		tmpLbl[:,:,0] = label[:,:,0]


		tmpImg = image.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}

import torch.nn.functional as F

class ResizeInterpolate(object):
    def __init__(self, size=(992, 256)):
        """
        Args:
            size (tuple): (height, width) — must be divisible by 32
        """
        self.size = size

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        # make sure shape is CHW (from ToTensorLab)
        if image.dim() == 2:
            image = image.unsqueeze(0)
        if label.dim() == 2:
            label = label.unsqueeze(0)

        # interpolate requires NCHW
        # Stack radargram into 3 channels [3,H,W]
   # [3,H,W]
        
      
      
        image = F.interpolate(image.unsqueeze(0).float(),
                              size=self.size, mode="bilinear",
                              align_corners=False).squeeze(0)
      
        image = image.repeat(3, 1, 1)  # repeat to 3 channels
    
       # label = F.interpolate(label.unsqueeze(0).float(), size=self.size, mode="nearest").squeeze(0)



        return {
            'imidx': imidx,
            'image': image,
            'label': label
        }

class SalObjDataset(Dataset):
	def __init__(self,img_name_list,lbl_name_list,transform=None):
		# self.root_dir = root_dir
		# self.image_name_list = glob.glob(image_dir+'*.png')
		# self.label_name_list = glob.glob(label_dir+'*.png')
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self,idx):

		# image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
		# label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])

		image= self.image_name_list[idx]
		imname = self.image_name_list[idx]
		imidx = np.array([idx])

		if(0==len(self.label_name_list)):
			label_3 = np.zeros(image.shape)
		else:
			label_3 = self.label_name_list[idx]


		label = np.zeros(label_3.shape[0:2])
		if(3==len(label_3.shape)):
			label = label_3[:,:,0]
		elif(2==len(label_3.shape)):
			label = label_3


		if(3==len(image.shape) and 2==len(label.shape)):
			label = label[:,:,np.newaxis]
		
		elif(2==len(image.shape) and 2==len(label.shape)):
			image = image[:,:,np.newaxis]
			label = label[:,:,np.newaxis]
		

		sample = {'imidx':imidx, 'image':image, 'label':label}

		if self.transform:
			sample = self.transform(sample)

		return sample

