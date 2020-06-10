import sys
import pandas as pd
import os
import numpy as np

header_list=sys.argv[1]
print("header_list = " + header_list)

df_header_list=pd.read_csv(header_list)
n_dir=len(df_header_list)
print("n of dir = " + str(n_dir))
print("shape of list = " + str(df_header_list.shape))

Tij_percentile_Conv=np.zeros([7,2])
Tij_percentile_DF=np.zeros([7,2])
#print("Tij_percentile = " + str(Tij_percentile.shape))

log_path=os.path.splitext(header_list)[0] + "_dir_info.csv"
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

    if(os.path.exists(Tij_csv_file)):
        df_Tij_csv_file=pd.read_csv(Tij_csv_file)
        #print("shape of Tij_csv = " + str(df_Tij_csv_file.shape))

        Tij_data=np.asarray(df_Tij_csv_file.iloc[1:,2],dtype="float32")
        #print(Tij_data.shape)

        Tij_percentile_C=np.percentile(Tij_data, q=[0,25,50,75,100])
        average=np.average(Tij_data)
        std=np.std(Tij_data)
        Tij_percentile_C=np.append(Tij_percentile_C,average)
        Tij_percentile_C=np.append(Tij_percentile_C,std)
        #print(Tij_percentile_C)

        #delta_Free
        dir=dir[0:len(dir)-8]
        Tij_csv_file_D=dir + "_final_rdens_Tij.csv"
        Tij_csv_file_D=os.path.join("G:\\DFPR\\",dir,Tij_csv_file_D)

        if(os.path.exists(Tij_csv_file_D)):
            df_Tij_csv_file_D=pd.read_csv(Tij_csv_file_D)
            Tij_data=np.asarray(df_Tij_csv_file_D.iloc[1:,2],dtype="float32")
            Tij_percentile_D=np.percentile(Tij_data, q=[0,25,50,75,100])
            average=np.average(Tij_data)
            std=np.std(Tij_data)
            Tij_percentile_D=np.append(Tij_percentile_D,average)
            Tij_percentile_D=np.append(Tij_percentile_D,std)
            #print(Tij_percentile_D)

            Tij_percentile_Conv=np.insert(Tij_percentile_Conv,0,Tij_percentile_C,axis=1)
            Tij_percentile_DF=np.insert(Tij_percentile_DF,0,Tij_percentile_D,axis=1)
            #print(Tij_percentile)
            #exit()
            #print(Tij_percentile.shape)

            log_text=str(n) + "," + dir + "\n"
            with open(log_path, mode='a') as log:
	            log.write(log_text)
            n=n+1

Tij_percentile_Conv=np.rot90(Tij_percentile_Conv)
Tij_percentile_DF=np.rot90(Tij_percentile_DF)
df_Tij_percentile_Conv=pd.DataFrame(Tij_percentile_Conv,columns=["0","25","50","75","100","average","std"])
df_Tij_percentile_DF=pd.DataFrame(Tij_percentile_DF,columns=["0","25","50","75","100","average","std"])
#for i in range(Tij_percentile_Conv.shape[0]): 
#    if(i%2==0):
#        df_Tij_percentile_Conv.rename(index={i:str(i//2+1)},inplace=True)
#    else:
#        df_Tij_percentile_Conv.rename(index={i:""},inplace=True)
#print(df_Tij_percentile)

foname=os.path.splitext(header_list)[0] + "_box_plot_Conv.csv"
df_Tij_percentile_Conv.to_csv(foname)
foname=os.path.splitext(header_list)[0] + "_box_plot_DF.csv"
df_Tij_percentile_DF.to_csv(foname)
