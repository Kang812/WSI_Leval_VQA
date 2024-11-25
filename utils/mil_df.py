import pandas as pd
import os
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def mil_df_gen(wsi_ids, N_category, patches_dir, train = True):
   
   paths = []
   wsis = []
   is_valids = []
   labels = []
   
   for i in tqdm(range(len(wsi_ids))):
    file_names = os.listdir(os.path.join(patches_dir, wsi_ids[i]))
    label = N_category[i]

    for file_name in file_names:
        path = os.path.join(patches_dir, wsi_ids[i], file_name)
        paths.append(path)
        wsis.append(wsi_ids[i])
        if train:
           is_valids.append(0)
        else:
           is_valids.append(1)
        
        if label == 0:
            labels.append("False")
        else:
           labels.append("True")
    
   df = pd.DataFrame({
       "path":paths,
       "wsi":wsis,
       "is_valid":is_valids,
       "label":labels})
    
   return df
   
        

if __name__ == '__main__':
    train_df = pd.read_csv("/workspace/whole_slide_image_LLM/data/ori_train.csv")
    train, valid = train_test_split(train_df, test_size = 0.2, stratify= train_df['N_category'], random_state=42)

    patches_dir = '/workspace/whole_slide_image_LLM/data/patches_dataset/train/'
    wsi_ids = train['ID'].to_list()
    N_category = train['N_category'].to_list()
    train_df = mil_df_gen(wsi_ids, N_category, patches_dir, train = True)
    
    wsi_ids = valid['ID'].to_list()
    N_category = valid['N_category'].to_list()
    valid_df = mil_df_gen(wsi_ids, N_category, patches_dir, train = False)

    df = pd.concat([train_df, valid_df])
    df.to_csv("/workspace/whole_slide_image_LLM/data/patches_dataset/train.csv", index = False)        