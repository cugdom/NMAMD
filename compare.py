# Calculate the similarity between org and adv and save the value to csv.
import csv
import os
import cv2
import numpy as np
import math
from skimage.measure import compare_nrmse	
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr	
from skimage.measure import shannon_entropy	

def SMD2(img):
    shape = np.shape(img)
    out = 0
    for y in range(0, shape[0]-1):
        for x in range(0, shape[1]-1):
            out+=math.fabs(int(img[x,y])-int(img[x+1,y]))*math.fabs(int(img[x,y]-int(img[x,y+1])))
    return out

def s_read_img(s_img_path,s_img_size=None):
	img_data = cv2.imread(s_img_path)
	if(s_img_size!=None):
		img_data = cv2.resize(img_data,s_img_size)
	return img_data

def s_img_compare(s_org_path,s_adv_path,s_default_size,s_adv_name):
	data_img_org = s_read_img(s_org_path,s_default_size)
	data_img_adv = s_read_img(s_adv_path)		
	nrmse = compare_nrmse(data_img_org,data_img_adv,norm_type='euclidean')	
	ssim = compare_ssim(data_img_org,data_img_adv,multichannel=True)
	psnr = compare_psnr(data_img_org,data_img_adv)	
	entropy = shannon_entropy(data_img_adv,base=2)	 
	img_adv_gray = cv2.cvtColor(data_img_adv.copy(),cv2.COLOR_BGR2GRAY)	
	i_Laplacian = cv2.Laplacian(img_adv_gray,cv2.CV_64F).var()		
	i_SMD2 = SMD2(img_adv_gray)	

	return [s_adv_name,nrmse,ssim,psnr,entropy,i_Laplacian,i_SMD2]	

# calculate the similarity between adversarial examples and the original images and save to csv.
def m_compare(m_org_dir,m_adv_root,m_tar_model,m_save_dir):	
	tar_model_name = m_tar_model[0]
	default_size = m_tar_model[1]

	adv_dir = os.path.join(m_adv_root,tar_model_name)    
	path_compare = os.path.join(m_save_dir,tar_model_name+"_img_compare.csv")		
	header = ['adv_name','nrmse','ssim','psnr','entropy','Laplacian','SMD2']
	body = []
	with open(path_compare,'w',newline='',encoding="utf-8") as f:
		f_csv = csv.writer(f)    
		f_csv.writerow(header)			

	adv_name_list = os.listdir(adv_dir)   
	for adv_name in adv_name_list:
		adv_path = os.path.join(adv_dir,adv_name)
		org_path = os.path.join(m_org_dir,adv_name[3:])	

		# Calculate six indicators
		list_indicator = s_img_compare(org_path,adv_path,default_size,adv_name)
		body.append(list_indicator)		

	# save to csv
	with open(path_compare,'a',newline='',encoding="utf-8") as f:
		f_csv = csv.writer(f)
		f_csv.writerows(body)
	print("success!")		

#====== global variable ======
g_org_dir = r"..\CODE_LAB\images\org"	
g_adv_root = r"..\CODE_LAB\images\adv"
g_save_dir = r'..\CODE_LAB\images\statistic'
g_tar_model = ["vgg19",(224,224)]		# tar_model: inception_v3  (299,299)、 resnet50 (224,224)、 vgg19 (224,224)、 xception (299,299)

#======	main function ======
m_compare(g_org_dir,g_adv_root,g_tar_model,g_save_dir)	




