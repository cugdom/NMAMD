# Algorithm 2: The Assessment of Model Generalization
import os
import pandas as pd
import json


def m_predict_compare(m_read_root,m_save_root,m_tar_model,m_pre_model):
	
	# Read the prediction from csv
	file_read_name = m_tar_model+'_adv_predicted_by_'+m_pre_model+'.csv'		
	org_predict_path = os.path.join(m_read_root,file_read_name)		
	org_predict_data = pd.read_csv(org_predict_path)

	d_comp_cw = pd.DataFrame(columns=['adv_name','preds_adv_class','preds_adv_accuracy','org_label','comp_results'])		
	d_comp_df = pd.DataFrame(columns=['adv_name','preds_adv_class','preds_adv_accuracy','org_label','comp_results'])	

	counter = 0
	num_cw_true = 0		
	num_df_true = 0

	num_cw_false = 0		
	num_df_false = 0

	for index,row in org_predict_data.iterrows():				
		adv_name = row['img_name']
		preds_adv_class = row['predict_class_id']
		preds_adv_accuracy = row['accuracy']
		org_label = row['img_name'].split('_')[1]		
		comp_results = (preds_adv_class == org_label)

		# Compare with CW
		if('CW_' in row['img_name']):
			d_comp_cw = d_comp_cw.append({
				'adv_name':adv_name,
				'preds_adv_class':preds_adv_class,
				'preds_adv_accuracy':preds_adv_accuracy,
				'org_label':org_label,
				'comp_results':comp_results
				},ignore_index=True)		
			if(comp_results):
				num_cw_true = num_cw_true + 1		
			if(not comp_results):
				num_cw_false = num_cw_false + 1		
		
		# Compare with DF 
		if('DF_' in row['img_name']):
			d_comp_df = d_comp_df.append({
				'adv_name':adv_name,
				'preds_adv_class':preds_adv_class,
				'preds_adv_accuracy':preds_adv_accuracy,
				'org_label':org_label,
				'comp_results':comp_results
				},ignore_index=True)	
			if(comp_results):
				num_df_true = num_df_true + 1			
			if(not comp_results):
				num_df_false = num_df_false + 1		

		counter = counter+1
		print('Number of executions: {}'.format(counter))					
	

	# Calculate the  prediction ratio of correct (True) and incorrect (False)
	cw_false_stat = {'Total adversarial examples predicted by cw algorithm':len(d_comp_cw),
										'True number of adversarial examples predicted by cw algorithm':num_cw_true,
										'False number of adversarial examples predicted by cw algorithm':num_cw_false,
										'True ratio':num_cw_true/len(d_comp_cw),
										'False ratio':num_cw_false/len(d_comp_cw)}
	df_false_stat = {'Total adversarial examples predicted by df algorithm':len(d_comp_df),
										'True number of adversarial examples predicted by df algorithm':num_df_true,
										'False number of adversarial examples predicted by df algorithm':num_df_false,
										'True ratio':num_df_true/len(d_comp_df),
										'False ratio':num_df_false/len(d_comp_df)}										

	file_save_cw_name = m_tar_model+'_adv_predicted_by_'+m_pre_model+'_cw.csv'			
	file_save_df_name = m_tar_model+'_adv_predicted_by_'+m_pre_model+'_df.csv'		
	stat_cw_save_path = os.path.join(m_save_root,m_tar_model+'_adv',file_save_cw_name)		
	stat_df_save_path = os.path.join(m_save_root,m_tar_model+'_adv',file_save_df_name)
	d_comp_cw.to_csv(stat_cw_save_path)
	d_comp_df.to_csv(stat_df_save_path)

	print('=== success !===')
	return [cw_false_stat,df_false_stat]



def m_loop(m_read_root,m_save_root):
	model_list = ['inception_v3','resnet50','vgg19','xception']
	false_stat = {}		
	for tar_model in model_list:	
		false_tar_model = {}	
		for pre_model in model_list:
			stat_rst = m_predict_compare(m_read_root,m_save_root,tar_model,pre_model)
			false_tar_model[pre_model] = stat_rst
		false_stat[tar_model] = false_tar_model

	# save the results
	stat_save_path = os.path.join(m_save_root,'predict_true_false_stat.json')	
	with open(stat_save_path,'w') as f:	
		json.dump(false_stat, f, indent = 4,ensure_ascii=False)		
	
	print('success')


###### global variable ####################
g_read_root = r'..\CODE_LAB\images\statistic\predict'	
g_save_root = r'..\CODE_LAB\images\statistic\compare'		
g_tar_model = 'inception_v3'		# Target model:  inception_v3		resnet50		vgg19   xception
g_pre_model = 'inception_v3'		# Prediction model:  inception_v3		resnet50		vgg19   xception

######	main function ####################
m_loop(g_read_root,g_save_root)

