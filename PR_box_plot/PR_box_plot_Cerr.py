import sys
import pandas as pd
import os
import numpy as np

def np_percentile(MPR_log_file):
    col_names=["trial", "iteration", "scale_factor", "Rfactor", "OS_ratio", "gamma", "NOR_ex"]
    df_MPR_log_file=pd.read_csv(MPR_log_file,sep=" ",names=col_names)
    index_trial=-1
    for i in range(df_MPR_log_file.shape[0]):
        if(df_MPR_log_file.iat[i,0] == "sta_dens"):
            n_trial=int(df_MPR_log_file.iat[i,5])
            
        if(df_MPR_log_file.iat[i,0] == "trial"):
            index_trial=i
            break 
    if(index_trial == -1):
        print("MPR_log file error")
    else:
        scale_factor=np.asarray(df_MPR_log_file.iloc[index_trial+1:index_trial+1+n_trial,2],dtype="float32")
        Rfactor=np.asarray(df_MPR_log_file.iloc[index_trial+1:index_trial+1+n_trial,3],dtype="float32")
        OS_ratio=np.asarray(df_MPR_log_file.iloc[index_trial+1:index_trial+1+n_trial,4],dtype="float32")
        gamma=np.asarray(df_MPR_log_file.iloc[index_trial+1:index_trial+1+n_trial,5],dtype="float32")
        Cerr=np.asarray(df_MPR_log_file.iloc[index_trial+1:index_trial+1+n_trial,6],dtype="float32")
        
        Rfactor_percentile=np.nanpercentile(Rfactor, q=[0,25,50,75,100])
        average=np.nanmean(Rfactor)
        std=np.nanstd(Rfactor)
        Rfactor_percentile=np.append(Rfactor_percentile,average)
        Rfactor_percentile=np.append(Rfactor_percentile,std)

        gamma_percentile=np.nanpercentile(gamma, q=[0,25,50,75,100])
        average=np.nanmean(gamma)
        std=np.nanstd(gamma)
        gamma_percentile=np.append(gamma_percentile,average)
        gamma_percentile=np.append(gamma_percentile,std)     

        Cerr_percentile=np.nanpercentile(Cerr, q=[0,25,50,75,100])
        average=np.nanmean(Cerr)
        std=np.nanstd(Cerr)
        Cerr_percentile=np.append(Cerr_percentile,average)
        Cerr_percentile=np.append(Cerr_percentile,std)        


        return Rfactor_percentile,gamma_percentile,Cerr_percentile,index_trial

def df_rot(percentile):
    percentile=np.rot90(percentile)
    df_percentile=pd.DataFrame(percentile,columns=["0","25","50","75","100","average","std"])
    return df_percentile

header_list=sys.argv[1]
print("header_list = " + header_list)

df_header_list=pd.read_csv(header_list)
n_dir=len(df_header_list)
print("n of dir = " + str(n_dir))
print("shape of list = " + str(df_header_list.shape))

Rfactor_percentile_Con=np.zeros([7,2])
Rfactor_percentile_DF=np.zeros([7,2])
gamma_percentile_Con=np.zeros([7,2])
gamma_percentile_DF=np.zeros([7,2])
Cerr_percentile_Con=np.zeros([7,2])
Cerr_percentile_DF=np.zeros([7,2])

log_path=os.path.splitext(header_list)[0] + "_dir_info_Cerr.csv"
log_text="num,dir_name" +"\n"
with open(log_path, mode='w') as log:
	log.write(log_text)

n=1
for i_dir in range(n_dir):
    dir=df_header_list.iat[i_dir,0]
    dir=dir[0:len(dir)-1]
    print("dir = " + dir)

    Tij_csv_file=dir + "_final_rdens_Tij.csv"
    Tij_csv_file=os.path.join("G:\\DFPR\\",dir,Tij_csv_file)
    #print("Tij_csv_file = " + Tij_csv_file)

    MPR_log_file=dir + "_MPR.log"
    MPR_log_file=os.path.join("G:\\DFPR\\",dir,MPR_log_file)
    #print(MPR_log_file)

    if(os.path.exists(Tij_csv_file)):
        Rfactor_percentile_C,gamma_percentile_C,Cerr_percentile_C,index_trial=np_percentile(MPR_log_file)
        if(index_trial != -1):

            #delta_Free
            dir=dir[0:len(dir)-8]
            Tij_csv_file_D=dir + "_final_rdens_Tij.csv"
            Tij_csv_file_D=os.path.join("G:\\DFPR\\",dir,Tij_csv_file_D)

            MPR_log_file=dir + "_MPR.log"
            MPR_log_file=os.path.join("G:\\DFPR\\",dir,MPR_log_file)

            if(os.path.exists(Tij_csv_file_D)):
                Rfactor_percentile_D,gamma_percentile_D,Cerr_percentile_D,index_trial=np_percentile(MPR_log_file)
                if(index_trial != -1):

                    Rfactor_percentile_Con=np.insert(Rfactor_percentile_Con,0,Rfactor_percentile_C,axis=1)
                    Rfactor_percentile_DF=np.insert(Rfactor_percentile_DF,0,Rfactor_percentile_D,axis=1)
                    gamma_percentile_Con=np.insert(gamma_percentile_Con,0,gamma_percentile_C,axis=1)
                    gamma_percentile_DF=np.insert(gamma_percentile_DF,0,gamma_percentile_D,axis=1)
                    Cerr_percentile_Con=np.insert(Cerr_percentile_Con,0,Cerr_percentile_C,axis=1)
                    Cerr_percentile_DF=np.insert(Cerr_percentile_DF,0,Cerr_percentile_D,axis=1)       

                    log_text=str(n) + "," + dir + "\n"
                    with open(log_path, mode='a') as log:
	                    log.write(log_text)
                    n=n+1                         

df_Rfactor_percentile_Con=df_rot(Rfactor_percentile_Con)
df_Rfactor_percentile_DF=df_rot(Rfactor_percentile_DF)
df_Cerr_percentile_Con=df_rot(Cerr_percentile_Con)
df_Cerr_percentile_DF=df_rot(Cerr_percentile_DF)
df_gamma_percentile_Con=df_rot(gamma_percentile_Con)
df_gamma_percentile_DF=df_rot(gamma_percentile_DF)

foname=os.path.splitext(header_list)[0] + "_Rfactor_Con.csv"
df_Rfactor_percentile_Con.to_csv(foname)
foname=os.path.splitext(header_list)[0] + "_Rfactor_DF.csv"
df_Rfactor_percentile_DF.to_csv(foname)

foname=os.path.splitext(header_list)[0] + "_Cerr_Con.csv"
df_Cerr_percentile_Con.to_csv(foname)
foname=os.path.splitext(header_list)[0] + "_Cerr_DF.csv"
df_Cerr_percentile_DF.to_csv(foname)

foname=os.path.splitext(header_list)[0] + "_gamma_Con.csv"
df_gamma_percentile_Con.to_csv(foname)
foname=os.path.splitext(header_list)[0] + "_gamma_DF.csv"
df_gamma_percentile_DF.to_csv(foname)
