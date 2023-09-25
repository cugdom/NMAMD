# NMAMD results analysis

import csv

# Calculate the accuracy of error correction
def m_mtd_recovery_stat(m_read_path):
	rst_total = 0
	true_total = 0
	row_key = []
	file_name = m_read_path.split('\\')[-1]
	with open(m_read_path,'r', encoding="utf-8") as f: 
		f_csv = csv.reader(f)     
		for row in f_csv:
			if(rst_total==0):		
				row_key = row
				rst_total = rst_total+1
				continue		
			row_data = dict(zip(row_key,row)) 	
			
			img_lable = row_data['image_name'].split('_')[1]		
			if(img_lable == row_data['arbiter_result'] and row_data['is_adv']=='yes'):		
				true_total = true_total+1
			rst_total = rst_total+1
	rst_total = rst_total-1	
	true_rate = true_total/rst_total
	return file_name,rst_total,true_total,true_rate



# Calculate the accuracy of the judgment as adv
def m_mtd_pre_adv_stat(m_read_path):
	rst_total = 0
	is_adv = 0
	row_key = []
	file_name = m_read_path.split('\\')[-1]
	with open(m_read_path,'r', encoding="utf-8") as f: 
		f_csv = csv.reader(f)      
		for row in f_csv:
			if(rst_total==0):			
				row_key = row
				rst_total = rst_total+1
				continue		
			row_data = dict(zip(row_key,row)) 	
			
			if(row_data['is_adv'] == 'yes'):
				is_adv = is_adv+1
			rst_total = rst_total+1
	rst_total = rst_total-1	
	pre_adv_rate = is_adv/rst_total
	return file_name,rst_total,is_adv,pre_adv_rate


# Calculate the accuracy of the judgment as adv(improvement: If one of the prediction accuracy is less than 50%, then judged as adversarial example)
def m_mtd_pre_adv_stat_imp(m_read_path):
	rst_total = 0
	is_adv = 0
	row_key = []
	file_name = m_read_path.split('\\')[-1]
	with open(m_read_path,'r', encoding="utf-8") as f: 
		f_csv = csv.reader(f)      
		for row in f_csv:
			if(rst_total==0):		
				row_key = row
				rst_total = rst_total+1
				continue		
			row_data = dict(zip(row_key,row)) 	
			
			if(row_data['is_adv'] == 'yes'):
				is_adv = is_adv+1
			else:
				if('accuracy_4' in row_data):
					if(float(row_data['accuracy_1'])<0.5 or float(row_data['accuracy_2'])<0.5 or float(row_data['accuracy_3'])<0.5 or float(row_data['accuracy_4'])<0.5):
						is_adv = is_adv+1
				elif('accuracy_3' in row_data):
					if(float(row_data['accuracy_1'])<0.5 or float(row_data['accuracy_2'])<0.5 or float(row_data['accuracy_3'])<0.5):
						is_adv = is_adv+1
				elif('accuracy_2' in row_data):
					if(float(row_data['accuracy_1'])<0.5 or float(row_data['accuracy_2'])<0.5):
						is_adv = is_adv+1
				elif('accuracy_1' in row_data):
					if(float(row_data['accuracy_1'])<0.5):
						is_adv = is_adv+1

			rst_total = rst_total+1
	rst_total = rst_total-1	
	pre_adv_rate = is_adv/rst_total
	return file_name,rst_total,is_adv,pre_adv_rate


#======	 global variable 	======
g_read_path = r'..\CODE_LAB\data\mtd\mtd_predict.csv'		
g_save_dir = r'..\CODE_LAB\data\mtd_result_stat'			

#======		main function 	======
m_mtd_recovery_stat(g_read_path)			# Calculate the accuracy of error correction
m_mtd_pre_adv_stat(g_read_path)			# Calculate the accuracy of the judgment as adv
m_mtd_pre_adv_stat_imp(g_read_path)			# Calculate the accuracy of the judgment as adv (improvement)