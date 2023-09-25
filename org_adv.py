# Calculate the difference in pixels between org and adv and save the value to csv.
import numpy as np
import scipy.misc
import os
import csv

# init csv . 
def s_init_csv(s_csv_fold,s_tar_model):
    csv_path = {}
    headers = ['org_name','index','gap']
    org_cw_csv_path = os.path.join(s_csv_fold,s_tar_model+"_org_cw.csv")
    org_df_csv_path = os.path.join(s_csv_fold,s_tar_model+"_org_df.csv")
    if not (os.path.exists(org_cw_csv_path)):      
        with open(org_cw_csv_path,'w',newline='',encoding="utf-8") as f:
            f_csv = csv.writer(f)    
            f_csv.writerow(headers)
    if not (os.path.exists(org_df_csv_path)):        
        with open(org_df_csv_path,'w',newline='',encoding="utf-8") as f:
            f_csv = csv.writer(f)    
            f_csv.writerow(headers)              
    csv_path["cw"] = org_cw_csv_path
    csv_path["df"] = org_df_csv_path
    return csv_path

def m_diff(m_org_dir,m_adv_root,m_tar_model,m_csv_dir):
    adv_dir = os.path.join(m_adv_root,m_tar_model)
    if (m_tar_model == "inceptionV3" or m_tar_model == "xception"):
        adv_size = (299,299)
    else:
        adv_size = (224,224)

    csv_path = s_init_csv(m_csv_dir,m_tar_model)  
    adv_list = os.listdir(adv_dir)   
    count = 0
    for adv_name in adv_list:
        adv_path = os.path.join(adv_dir,adv_name)
        org_name = adv_name[3:]
        org_path = os.path.join(m_org_dir,org_name)     

        org_img = scipy.misc.imread(org_path)
        adv_img = scipy.misc.imread(adv_path)
        
        org_data = np.array(scipy.misc.imresize(org_img, adv_size),dtype=np.float32) 
        org_shape = org_data.shape

        adv_diff_cw = []
        adv_diff_df = []
        for i in range(org_shape[2]):
            for j in range(org_shape[1]):
                for k in range(org_shape[0]):
                    if('CW' in adv_name):       
                      adv_diff_cw.append((org_name,(k,j,i),adv_img[k,j,i] - org_data[k,j,i]))      
                    if('DF' in adv_name):
                      adv_diff_df.append((org_name,(k,j,i),adv_img[k,j,i] - org_data[k,j,i]))     

        # add to csv
        with open(csv_path['cw'],'a',newline='',encoding="utf-8") as f:
            f_csv = csv.writer(f)
            f_csv.writerows(adv_diff_cw)
        with open(csv_path['df'],'a',newline='',encoding="utf-8") as f:
            f_csv = csv.writer(f)
            f_csv.writerows(adv_diff_df)

        count =count+1
        print('比较图片数为{}'.format(count))          
       
    print("===【success !】======")


#======	global variable	======
g_org_dir = r'..\CODE_LAB\images\org'    
g_adv_root = r'..\CODE_LAB\images\adv'   
g_tar_model = 'xception'        # inception_v3   resnet50    vgg19    xception
g_save_dir = r'..\CODE_LAB\images\statistic'   #  the save path for csv

#======	main function ======
m_diff(g_org_dir,g_adv_root,g_tar_model,g_save_dir)

