import pandas as pd
import os
from sklearn.model_selection import train_test_split


def data_split(df_path, data_dir, save_path):
    df = pd.read_csv(df_path)
    df = df[['img_path', 'N_category']]
    filenames = [os.path.join(data_dir,i.split("/")[-1]) for i in df['img_path'].to_list()]
    df['img_path'] = filenames
    df.columns = ['image_path', 'label']
    train, valid = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    train.to_csv(os.path.join(save_path, 'train.csv'), index = False)
    valid.to_csv(os.path.join(save_path, 'valid.csv'), index = False)



if __name__ == '__main__':
    df_path = '/workspace/whole_slide_image_LLM/data/train.csv'
    data_dir = '/workspace/whole_slide_image_LLM/data/image/train/'
    save_path = '/workspace/whole_slide_image_LLM/data/image/'
    data_split(df_path, data_dir, save_path)