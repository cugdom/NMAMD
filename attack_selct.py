# target model:vgg16    target algorithm:en generate adversarial examples for NMAMD
import numpy as np
import keras
from keras import backend
from keras.models import load_model
import tensorflow as tf
from cleverhans.utils_keras import KerasModelWrapper   
from keras.applications import vgg16
from keras.applications import mobilenet
from keras.applications import imagenet_utils
import cleverhans.attacks
import scipy.misc
import os
import csv
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


# attack algorithm
def s_adv_en(s_img_data,s_model,s_sess,s_adv_path,s_adv_num):
	l_tar_model = KerasModelWrapper(s_model)	 
	l_img_data = np.array([s_img_data / 255.0])	

	adv_obj = cleverhans.attacks.ElasticNetMethod(l_tar_model, sess = s_sess)	
	adv_params = { 'max_iterations':1000,    
									'confidence':10,
									'initial_const':1e-3
								}     
	adv = adv_obj.generate_np(l_img_data, **adv_params)
	scipy.misc.imsave(s_adv_path, adv[0])  
	s_adv_num = s_adv_num + 1  
	print('EN generate the {} adversarial example'.format(s_adv_num))  
	return s_adv_num     

def s_read_csv(s_csv_path):
	org_img_path = []
	with open(s_csv_path,'r', encoding="utf-8") as f: 
		f_csv = csv.reader(f)     
		for row in f_csv:
			org_img_path.append(row)
	return org_img_path

def s_create_model(s_model):
  if(s_model=='vgg16'):
    obj_model = vgg16.VGG16(weights='imagenet')
    default_size = (224, 224)
  return obj_model,default_size



# Select effective original images
def m_predict_image(m_img_dir,m_save_root,m_model):	
	save_dir = os.path.join(m_save_root,'predicted_clean')
	p_model_obj = vgg16.VGG16(weights='imagenet')	
	i_name_list = os.listdir(m_img_dir)		
	i_sum = len(i_name_list)     
	i_name_list_random= np.random.choice(i_name_list,size=i_sum,replace=False)		

	counter = 1    	
	sect_dict = {}
	for i_name in i_name_list_random:
		i_path = os.path.join(m_img_dir,i_name)		
		img = load_img(i_path,target_size=m_model[1])
		img_data = img_to_array(img)
		img_batch = np.expand_dims(img_data, axis=0)   # image_batch.shape = (1,224,224,3)  
		img_pre_data = mobilenet.preprocess_input(img_batch.copy())
		img_pre_data = vgg16.preprocess_input(img_batch.copy())
		preds = p_model_obj.predict(img_pre_data)
		rst = imagenet_utils.decode_predictions(preds) 
		r_label = i_name.split('_')[0]	# real label
		p_label = rst[0][0][0]		# predict label
		counter = counter+1
		if(r_label==p_label and i_name not in sect_dict.keys() and rst[0][0][2]>0.8):	
			sect_dict[i_name] = rst[0][0]
			print("The number of the prediction: {}".format(len(sect_dict)))
		if(len(sect_dict)==500):		
			break

	save_path = os.path.join(save_dir,m_model[0]+'_predict.csv')	
	with open(save_path,'a+',newline='') as f:			
		f_csv = csv.writer(f)
		for row in sect_dict:			
			f_csv.writerow([row,sect_dict[row][0],sect_dict[row][1],sect_dict[row][2]])


# generate adversarial examples
def m_adv_generate(m_img_dir,m_save_root,m_tar_model):
	img_clean_dir = os.path.join(m_save_root,'predicted_clean')   
	adv_save_dir = os.path.join(m_save_root,'adv')
	img_clean_list = s_read_csv(os.path.join(img_clean_dir,m_tar_model[0]+'_predict.csv'))    
	file_sum = len(img_clean_list)     

	num_adv_sum = len(os.listdir(adv_save_dir))   

	backend.set_learning_phase(False)		
	tf.set_random_seed(1234)	
	if keras.backend.image_dim_ordering() != 'tf':
			keras.backend.set_image_dim_ordering('tf')    
	sess = backend.get_session()	

	# Create  model
	l_pre_model = vgg16.VGG16(weights='imagenet')

	# generate adversarial examples
	for img_clean in img_clean_list:  
		org_name = img_clean[0]	
		org_path = os.path.join(m_img_dir,org_name)      
		org_img = scipy.misc.imread(org_path)	
		if(len(org_img.shape)!=3):    
			continue
		org_img_data = np.array(scipy.misc.imresize(org_img, m_tar_model[1]),dtype=np.float32)  
		adv_save_path = os.path.join(adv_save_dir,"EN_"+org_name) 
		if(os.path.isfile(adv_save_path)):	
			continue   
		num_adv_sum = s_adv_en(org_img_data,l_pre_model,sess,adv_save_path,num_adv_sum)  

	print('=== success !===')

# Select effective adversarial examples
def m_predict_adv(m_save_root):
	adv_dir = os.path.join(m_save_root,'adv')  

	rst_save_path = os.path.join(m_save_root,'vgg16_predict_adv.csv')	
	headers = ['img_name','img_label','adv_label','adv_accy' ]			# the header of csv 
	with open(rst_save_path,'w',newline='',encoding="utf-8") as f :  
		f_writer = csv.writer(f)
		f_writer.writerow(headers)

	# create model 
	obj_model = vgg16.VGG16(weights='imagenet')    
	adv_name_list = os.listdir(adv_dir)
	body = []
	for adv_name in adv_name_list:
		adv_path = os.path.join(adv_dir,adv_name)
		adv = load_img(adv_path, target_size=(224,224))     
		adv_numpy = img_to_array(adv)    # convert to in PIL format
		adv_data = np.expand_dims(adv_numpy, axis=0)   
		adv_prep = vgg16.preprocess_input(adv_data.copy())    
		rst = obj_model.predict(adv_prep)
		lbl = imagenet_utils.decode_predictions(rst) 
		img_lable = adv_name.split('_')[1]
		if(img_lable==lbl[0][0][0]):
			print('The prediction label of the adversarial example is the same as the original: {}'.format([adv_name,lbl[0][0]])) 
		if(img_lable!=lbl[0][0][0]):		
			body.append([adv_name,img_lable,lbl[0][0][0],lbl[0][0][2]])
			
	with open(rst_save_path,'a',newline='',encoding="utf-8") as f:
		f_csv = csv.writer(f)
		for row in body:
			f_csv.writerow(row)

	print('=== success !===')


#======	 global variable 	======
g_img_dir = r"..\data\images\org"        
g_save_root = r'..\CODE_LAB\data\attack_selct'
g_tar_model = ['vgg16',(224, 224)]			


#======		main function 	======
# m_predict_image(g_img_dir,g_save_root,g_tar_model)
# m_adv_generate(g_img_dir,g_save_root,g_tar_model)	
m_predict_adv(g_save_root)				
