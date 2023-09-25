# Algorithm 3 NMAMD
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.applications import inception_v3
from keras.applications import vgg19
from keras.applications import resnet50
from keras.applications import xception
import numpy as np
import os
import csv
import pandas as pd
from collections import Counter
from keras import backend
import datetime

def save_img_list(i_fold,f_save):
	file_name_org = os.listdir(i_fold)  
	with open(f_save,'w',newline='',encoding="utf-8") as f:
		f_csv = csv.writer(f)
		for i_name in file_name_org:
			f_csv.writerow(i_name.split(' '))	
	print("=== success !===")


# Image preprocessing
def s_preprocess_input(m_name,image_data):	
	if(m_name=='inception_v3'):
		pro_image = inception_v3.preprocess_input(image_data)
	elif(m_name=='resnet50'):
		pro_image = resnet50.preprocess_input(image_data)
	elif(m_name=='vgg19'):
		pro_image = vgg19.preprocess_input(image_data)
	elif(m_name=='xception'):
		pro_image = xception.preprocess_input(image_data)
	return pro_image

			
# Scheduler: Selects the master model and the corresponding generalization list, and returns the selected alternative model entity
def s_scheduler(m_major_model_prob,m_units_order,m_opt,g_exe_number):		
	list_major_model=list(m_major_model_prob.keys())		# A list of names for the master model
	list_model_prob=list(m_major_model_prob.values())	  # List of probabilities corresponding to the selection of the master model
	
	if(m_opt=='r'):   # select execution body randomly 
		sect_major_model = np.random.choice(a=list_major_model,size=g_exe_number,replace=False)  
		exe_units = list(sect_major_model)		
	
	elif(m_opt=='p'):   # Choose the master model according to probability
		sect_major_model = np.random.choice(a=list_major_model,size=1,replace=False,p=list_model_prob)   
		sect_major_model = str(sect_major_model[0])		

		exe_units = m_units_order[sect_major_model].copy()	
		exe_units.insert(0,sect_major_model)	
		if(len(exe_units)<g_exe_number):		
			raise Exception('The number of executors must be less than the total number of executors')	
		exe_units = exe_units[0:g_exe_number]		

	else:
		raise Exception('Execution body selection setting error!')
	


	# Create the executors  from the executors  list
	m_list = []			# executors list
	for m_name in exe_units:  
		if(m_name=='inception_v3'):
			m_list.append(['inception_v3',inception_v3.InceptionV3(weights='imagenet'),(299,299)]) 
		elif(m_name=='resnet50'):
			m_list.append(['resnet50',resnet50.ResNet50(weights='imagenet'),(224,224)])
		elif(m_name=='vgg19'):
			m_list.append(['vgg19',vgg19.VGG19(weights='imagenet'),(224,224)])
		elif(m_name=='xception'):
			m_list.append(['xception',xception.Xception(weights='imagenet'),(299,299)])	
	print("Scheduling success!")
	return m_list

# Arbiter: Detect whether it is a countersample; 2, the majority of votes output the final result
def s_majority_vote(s_values):
	val_stat = Counter(s_values)	
	
	# Determine whether it is an adversarial example
	if(len(val_stat)!=1):		
		is_adv = 'yes'
	else:
		is_adv = 'no'

	val_counts = list(val_stat.values())
	count_max = max(val_counts)			
	keys_count_max = []	
	for k in val_stat:	
		if (val_stat[k] ==count_max):		
			keys_count_max.append(k)			
	if(len(keys_count_max)!=1):
		print('The result of the vote selected multiple values: {}, the vote failed.'.format(keys_count_max))
		vote_result = 0
	else:
		print('The vote is successful, and the result is: {}'.format(keys_count_max))
		vote_result = keys_count_max[0]
	return 	[is_adv,vote_result]


# NMAMD(contain :scheduler, executor, arbiter)
def m_mimesis_defender(m_img_root,m_save_dir,m_major_model_prob,m_units_order,m_opt,m_exe_number):
 
	adv_csv_path = os.path.join(m_img_root,'vgg16_predict_adv.csv')
	adv_dir = os.path.join(m_img_root,'adv')	
	adv_name_list = []
	with open(adv_csv_path) as f:
		adv_info = csv.reader(f)
		for row in adv_info:
			adv_name_list.append(row[0])
	del(adv_name_list[0])		

	# scheduler
	model_list = s_scheduler(m_major_model_prob,m_units_order,m_opt,m_exe_number)		

	timestamp = datetime.datetime.now().strftime('_%Y_%m_%d')
	rst_save_path = os.path.join(m_save_dir,'mtd_predict_rst_'+ m_opt+str(m_exe_number) + timestamp+'.csv')	
	
	# the header of csv
	if(m_exe_number==1):
		headers = ['image_name',
						'model_name_1','class_id_1','accuracy_1',
						'is_adv','arbiter_result' ]
	if(m_exe_number==2):
		headers = ['image_name',
						'model_name_1','class_id_1','accuracy_1',
						'model_name_2','class_id_2','accuracy_2',
						'is_adv','arbiter_result' ]
	if(m_exe_number==3):
		headers = ['image_name',
						'model_name_1','class_id_1','accuracy_1',
						'model_name_2','class_id_2','accuracy_2',
						'model_name_3','class_id_3','accuracy_3',
						'is_adv','arbiter_result' ]
	if(m_exe_number==4):				
		headers = ['image_name',
							'model_name_1','class_id_1','accuracy_1',
							'model_name_2','class_id_2','accuracy_2',
							'model_name_3','class_id_3','accuracy_3',
							'model_name_4','class_id_4','accuracy_4',
							'is_adv','arbiter_result' ]
	with open(rst_save_path,'w',newline='',encoding="utf-8") as f : 
		f_writer = csv.writer(f)
		f_writer.writerow(headers)

	flag = 0

	for adv_name in adv_name_list:		
		img_path = os.path.join(adv_dir,adv_name)		
		body = []   # save the data
		vote_candidates = []	# save the prediction
		body.append(adv_name)	
	
		for m in model_list:				
			img = load_img(img_path, target_size=m[2])				
			numpy_image = img_to_array(img)
			image_batch = np.expand_dims(numpy_image, axis=0)
			processed_image = s_preprocess_input(m[0],image_batch.copy())

			preds = m[1].predict(processed_image)			# executor
			P = imagenet_utils.decode_predictions(preds)
			body.append(m[0])	
			body.append(P[0][0][0])		
			body.append(P[0][0][2])		
			vote_candidates.append(P[0][0][0])		
		
		vote_result = s_majority_vote(vote_candidates)			# arbiter
		body.append(vote_result[0])			
		body.append(vote_result[1])		

		with open(rst_save_path,'a',newline='',encoding="utf-8") as f:
			f_csv = csv.writer(f)
			f_csv.writerow(body)
		body.clear()		
		backend.clear_session()		

		model_list = s_scheduler(m_major_model_prob,m_units_order,m_opt,g_exe_number)		
		flag = flag+1


	print(" Voting success!")

#======	 global variable 	======
g_adv_root = r'..\CODE_LAB\data\attack_selct'		
g_save_dir = r'..\CODE_LAB\data\mtd'

# Scheduling Policy
g_major_model_prob = {'inception_v3':0.275,'resnet50':0.272,'vgg19':0.258,'xception':0.195}		# from predict_compare
g_units_order ={'inception_v3':['xception','vgg19','resnet50'],		# from the True_rate of predict_compare
								'resnet50':['inception_v3','xception','vgg19'],
								'vgg19':['inception_v3','xception','resnet50'],
								'xception':['inception_v3','vgg19','resnet50'] }
g_opt ='p'		# 'r'：random		'p' ：probability
g_exe_number = 3				# executor number：3, 4


#======		main function 	======
m_mimesis_defender(g_adv_root,g_save_dir,g_major_model_prob,g_units_order,g_opt,g_exe_number)
