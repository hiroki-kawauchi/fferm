import glob
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def face_clustering(arr_encodings, n_clusters, imgdir_path):
    fr_path = os.path.join(imgdir_path,'fr_index')
    cl_path = os.path.join(imgdir_path,'cl_index')
    df_enc = pd.DataFrame(arr_encodings)

    kmeans_model = KMeans(n_clusters=n_clusters, random_state=10).fit(df_enc)
    labels = kmeans_model.labels_ + 1

    files = glob.glob(fr_path + '/*')
    l_cl = [[i+1,label]for i,label in enumerate(labels)]
    label_out = []

    os.makedirs(cl_path, exist_ok=True)

    for i,label in enumerate(labels):
        if not label in label_out:
            shutil.copy(files[i],os.path.join(cl_path,'{}.jpg'.format(label+1)))
            label_out.append(label)

    return l_cl

def fr_change_index(df_fr_raw, csvIndexToFixIndexDic):
    all_face_nums = df_fr_raw['face_number'].unique().tolist()
    not_drop = pd.DataFrame(list(csvIndexToFixIndexDic.items()), columns=[
                            'fr_index', 'cl_index'])
    not_drop = not_drop[not_drop['cl_index'] != 0]
    to_drop = [i for i in all_face_nums if not i in not_drop['fr_index'].unique(
    ).tolist()]  # allから消すもの
    df_replaced = df_fr_raw[~df_fr_raw['face_number'].isin(to_drop)]
    df_replaced = df_replaced.replace({'face_number': csvIndexToFixIndexDic})

    return df_replaced
