import pandas as pd
import os
from fferm import fferm,clustering

cwd = os.getcwd()
MOVIE_PATH = 'example.mp4'
OUTPUT_PATH = '{}/face_images'.format(cwd)

output_list,  encodings = fferm.fferm(MOVIE_PATH, OUTPUT_PATH)
df_output = pd.DataFrame(output_list)
df_output.columns = ['total time', 'x', 'y','width', 'height',
                    'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise',
                    'face_number']

N_CLUSTERS = 6 #num of people in the movie
l_cl = clustering.face_clustering(encodings, N_CLUSTERS, OUTPUT_PATH)

df_replaced = clustering.fr_change_index(df_output, dict(l_cl))
df_replaced.to_csv(os.path.join(cwd, 'fferm_clustering.csv'),index=False)
