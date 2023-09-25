# Data statistics and analysis
import pandas as pd	
import json
import os

# Read the file path by model or by adversarial example generation algorithm
def s_data_path_list(s_opt,s_stat_data_dir):
	f_path =[]	
	if(s_opt[0]=='M'):			#  read data by model
		path_img_diff_cw_stat = os.path.join(s_stat_data_dir,s_opt[1]+'_cw_diff_cmp_stat.json')   
		path_img_diff_df_stat = os.path.join(s_stat_data_dir,s_opt[1]+'_df_diff_cmp_stat.json')   
		f_path = [path_img_diff_cw_stat,path_img_diff_df_stat]
	if(s_opt[0]=='A'):		# read data by algorithm
		f_name = s_opt[1]+'_diff_cmp_stat.json'	
		path_img_diff_i_stat = os.path.join(s_stat_data_dir,'inception_v3_'+f_name)	
		path_img_diff_r_stat = os.path.join(s_stat_data_dir,'resnet50_'+f_name)		# resnet50
		path_img_diff_v_stat = os.path.join(s_stat_data_dir,'vgg19_'+f_name)			# vgg19
		path_img_diff_x_stat = os.path.join(s_stat_data_dir,'xception_'+f_name)		# xception
		f_path = [path_img_diff_i_stat,path_img_diff_r_stat,path_img_diff_v_stat,path_img_diff_x_stat]
	return f_path


# Data statistics and analysis
def m_stat_process(m_stat_data_dir,m_save_dir):
	r_name_list = os.listdir(m_stat_data_dir)			

	for r_name in r_name_list:
		s_name = r_name.split('.')[0]+'.csv'	
		r_path = os.path.join(m_stat_data_dir,r_name)			
		with open(r_path,'r') as f:
			json_stat_data = json.load(f)	

		imgs_stat = pd.DataFrame()
		pd.set_option('display.float_format',lambda x:'%.3f'%x)	

			
		for i in range(len(json_stat_data['img_name'])):
			j_name = json_stat_data['img_name'][str(i)]	
			j_data = pd.DataFrame.from_dict(json_stat_data['data'][str(i)])	
			j_data.columns = [j_name]	
			j_data = j_data.stack().unstack(0)		
			j_data = j_data.drop(['count'],axis=1)
			j_data = j_data.drop(['SMD2'],axis=1)	
			j_data['range'] = j_data['max']-j_data['min']	
			imgs_stat = imgs_stat.append(j_data)	
	
		save_dir = os.path.join(m_save_dir,s_name)	
		imgs_stat.to_csv(save_dir)

	print("success !")

###### global variable ####################
g_stat_data_dir = r'F:\CODE_LAB\images\statistic'
g_save_dir = r'F:\CODE_LAB\images\statistic\save'


######	main function ####################
m_stat_process(g_stat_data_dir,g_save_dir)		# tar_model: inceptionV3 resnet50 vgg19	xception	tar_adv: 'cw'	'df' 


