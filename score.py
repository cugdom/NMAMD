# Calculate score
import pandas as pd		
import os

# score the algorithms
def main_score_adv(fold_file,file_name,fold_save):
	file_path= os.path.join(fold_file,file_name)	
	d_cw = pd.read_csv(file_path)	
	d_cw = d_cw.drop(['Unnamed: 0','25%','50%','75%'],axis=1) 

	d_cw = d_cw.abs()	
	d_max = d_cw.max()
	d_min = d_cw.min()
	d_step = (d_max - d_min)/10	

	list_index = d_max.index	
	for i in range(0,len(list_index)):
		i_name = list_index[i]
		i_min = d_min[i_name]
		i_max = d_max[i_name]
		i_range = d_step[i_name]
		i_bins = [i_min-1,i_min+i_range,i_min+2*i_range,i_min+3*i_range,i_min+4*i_range,i_min+5*i_range,i_min+6*i_range,i_min+7*i_range,i_min+8*i_range,i_min+9*i_range,i_max+1]		
		i_labels = [10,9,8,7,6,5,4,3,2,1]	
		if(i_name=='Laplacian' or i_name=='entropy' or i_name=='psnr' or i_name=='ssim'):
			i_labels = [1,2,3,4,5,6,7,8,9,10]
		d_cw['s_'+i_name] = pd.cut(x=d_cw[i_name],bins = i_bins,labels=i_labels)

	path_save_file = os.path.join(fold_save,'S_10_'+file_name)	
	d_cw.to_csv(path_save_file)

	print("success !")

#  score the models
def s_score_model(s_fold_file,s_file_name,s_fold_save):
	file_path= os.path.join(s_fold_file,s_file_name)	
	d_cw = pd.read_csv(file_path)	
	d_cw = d_cw.drop(['Unnamed: 0','25%','50%','75%'],axis=1) 
	d_max = d_cw.max()
	d_min = d_cw.min()


	list_index = d_max.index	
	for i in range(0,len(list_index)):
		i_name = list_index[i]
		i_min = d_min[i_name]
		i_max = d_max[i_name]
		if(i_min*i_max<0):	
			i_max = max(i_max,abs(i_min))		
			i_min = 0				
			d_cw[i_name] = abs(d_cw[i_name])
		if(i_min<0 and i_max<0):	
			d_cw[i_name] = abs(d_cw[i_name])
			tmp = abs(i_min)		
			i_min = abs(i_max)
			i_max = tmp			

		i_range = (i_max-i_min)/10
		i_bins = [i_min-1,i_min+i_range,i_min+2*i_range,i_min+3*i_range,i_min+4*i_range,i_min+5*i_range,i_min+6*i_range,i_min+7*i_range,i_min+8*i_range,i_min+9*i_range,i_max+1]		
		i_labels = [1,2,3,4,5,6,7,8,9,10]	
		if(i_name=='Laplacian' or i_name=='entropy' or i_name=='psnr' or i_name=='ssim'):	
			i_labels = [10,9,8,7,6,5,4,3,2,1]
		print(i_name)
		d_cw['s_'+i_name] = pd.cut(x=d_cw[i_name],bins = i_bins,labels=i_labels)	 

	path_save_file = os.path.join(s_fold_save,'S_10_'+s_file_name)	
	d_cw.to_csv(path_save_file)

	print("success !")


def m_loop_score_calc(m_fold_file,m_fold_save):
	file_name_list = os.listdir(m_fold_file)
	for file_name in file_name_list:
		s_score_model(m_fold_file,file_name,m_fold_save)


###### global variable ####################
g_read_dir = r"F:\CODE_LAB\images\test_Aug\statistic\VII_main_stat_process\v1"	
g_save_dir = r'F:\CODE_LAB\images\test_Aug\statistic\IX_main_score_10\v1'	
g_data_file = 'inception_v3_cw_diff_cmp_stat.csv'	

######	main function ####################
m_loop_score_calc(g_read_dir,g_save_dir)
