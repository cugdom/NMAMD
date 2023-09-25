# Algorithm 1: Model Assessment
import pandas as pd
import json
import os

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  


# calculate the pixel difference and similarity
def m_diff_compare_stats(m_stat_root,m_model,m_adv_root,m_save_dir):
		
	diff_dir = os.path.join(m_stat_root,'II_main_org_adv')	
	diff_path_cw = os.path.join(diff_dir,m_model+'_org_cw.csv')
	diff_path_df = os.path.join(diff_dir,m_model+'_org_df.csv')

	compare_dir = os.path.join(m_stat_root,'statistic')	
	compare_path = os.path.join(compare_dir,m_model+'_img_compare.csv')


	adv_dir = os.path.join(m_adv_root,m_model)	
	adv_name_list = os.listdir(adv_dir)   

	#  read difference value for pixels  
	pd.set_option('display.float_format',lambda x:'%.3f'%x)	
	d_cw = pd.read_csv(diff_path_cw)	
	d_df = pd.read_csv(diff_path_df)

	# read similarity value
	d_cmpa = pd.read_csv(compare_path)	

	img_diff_cw_stat = pd.DataFrame(columns=['img_name','data'])
	img_diff_df_stat = pd.DataFrame(columns=['img_name','data'])	

	for adv_name in adv_name_list:
		if('CW' in adv_name): 
			org_name = adv_name[3:]
			
			img_diff_cw = d_cw[d_cw['org_name'].isin([org_name])]		
			stat_cw = img_diff_cw.describe()	
			stat_cw.loc['0_rate'] = [img_diff_cw['gap'].value_counts()[0]/img_diff_cw.count()['gap']]	

			img_cmpa_cw = d_cmpa[d_cmpa['adv_name'].isin([adv_name])].stack().unstack(0)	
			img_cmpa_cw = img_cmpa_cw.drop(['adv_name'],axis=0)	
			img_cmpa_cw.columns = ['gap']	
			stat_cw = stat_cw.append(img_cmpa_cw)	
			
			img_diff_cw_stat = img_diff_cw_stat.append({'img_name':org_name,'data':stat_cw.to_dict(orient='dict')},ignore_index=True)		# Save statistics value

		if('DF' in adv_name): 
			org_name = adv_name[3:]		
			img_diff_df = d_df[d_df['org_name'].isin([org_name])]					
			stat_df = img_diff_df.describe()			
			stat_df.loc['0_rate'] = [img_diff_df['gap'].value_counts()[0]/img_diff_df.count()['gap']]			
			img_cmpa_df = d_cmpa[d_cmpa['adv_name'].isin([adv_name])].stack().unstack(0)			
			img_cmpa_df = img_cmpa_df.drop(['adv_name'],axis=0)		
			img_cmpa_df.columns = ['gap']				
			stat_df = stat_df.append(img_cmpa_df)			
			img_diff_df_stat = img_diff_df_stat.append({'img_name':org_name,'data':stat_df.to_dict(orient='dict')},ignore_index=True)		
			
	path_img_diff_cw_stat = os.path.join(m_save_dir, m_model+'_cw_diff_cmp_stat.json')		
	path_img_diff_df_stat = os.path.join(m_save_dir, m_model+'_df_diff_cmp_stat.json')
	
	with open(path_img_diff_cw_stat,'w') as f:		
		json.dump(img_diff_cw_stat.to_dict(orient='dict'), f, indent = 4)		
	with open(path_img_diff_df_stat,'w') as f:	
		json.dump(img_diff_df_stat.to_dict(orient='dict'), f, indent = 4)		

	print("success!")	

###### global variable ####################
g_stat_root = r"..\CODE_LAB\images\statistic"
g_model = 'inception_v3'		# inception_v3 resnet50 vgg19  xception
g_adv_root = r'..\CODE_LAB\images\adv'
g_save_dir = r'..\CODE_LAB\images\statistic'

######	main function ####################
m_diff_compare_stats(g_stat_root,g_model,g_adv_root,g_save_dir)
