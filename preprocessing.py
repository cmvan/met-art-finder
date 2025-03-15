import os, os.path
import pandas as pd
import urllib.request
import re
import string
import concurrent.futures
import requests 
from PIL import Image 
from urllib.parse import urlparse
from pathlib import Path
from nltk.metrics import distance
import scipy.spatial as spatial
import numpy as np
from scipy.cluster.vq import kmeans
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import gensim.downloader
import re
import inflect
from sklearn.preprocessing import LabelEncoder

# def save_image_from_url(row, output_folder):
#     print(row.imageURL)
#     image = requests.get(row.imageURL)
#     quit()
#     name =  row.title



#     output_path = os.path.join(
#         output_folder, name
#     )
#     with open(output_path, "wb") as f:
#         f.write(image.content)

# def load(df, output_folder):    
#     with concurrent.futures.ThreadPoolExecutor(
#         max_workers=5
#     ) as executor:
#         future_to_url = {
#             executor.submit(save_image_from_url, row, output_folder): row
#             for index, row in df.iterrows()
#         }
#         for future in concurrent.futures.as_completed(
#             future_to_url
#         ):
#             url = future_to_url[future]
#             try:
#                 future.result()
#             except Exception as exc:
#                 print(
#                     "%r generated an exception: %s" % (url, exc)
#                 )

def move():
    DIR1 = './train'
    DIR2 = './val'

    val = False

    for name in os.listdir(DIR2):
        file_names = [x for x in os.listdir(os.path.join(DIR2,name))]

        for f in file_names:
            if f[-3:] != "jpg":
                print("rar")
                os.remove(os.path.join(DIR2, name, f))

def download():
    sub = pd.read_csv('train.csv')

    pattern = re.compile('[\W_]+')
    total = 0

    for i in range(0, sub.shape[0]):
        title = sub.iloc[i]["title"]
        # classification = sub.iloc[i]["classification"]
        url = sub.iloc[i]["imageURL"]

        path = urlparse(url).path
        ext = os.path.splitext(path)[1]
        folder = "merged_museum_images"
        
        # Path(folder).mkdir(parents=True, exist_ok=True)

        try:
            urllib.request.urlretrieve(url, f"{folder}/{title}.{ext}")
            total += 1
            print(i, title)
        except:
            print("request failed for", title)
    
    print(f"acquired {total} images out of", sub.shape[0])


def valid_files():
    sub = pd.read_csv('merged_museum_valid_links.csv')


    file_path = os.path.join("C:/Users/nbola/Downloads/cs229/src/final_data/merged_museum_images/", "Great Walls: The Great Wall of China") + "..jpg"

    print(file_path)
    
    if os.path.getsize(file_path) == 0:
        print("wow")
    
    total = 0
    remove = []

    for i in range(0, sub.shape[0]):
        title = sub.iloc[i]["title"]
        url = sub.iloc[i]["imageURL"]

        path = urlparse(url).path
        ext = os.path.splitext(path)[1]
        folder = "C:/Users/nbola/Downloads/cs229/src/final_data/merged_museum_images/"

        file_path = os.path.join(folder, title) + "..jpg"
        

        if not os.path.exists(file_path):
            total += 1
            print(total, title)
            remove.append(i)
    
    sub.drop(index=remove, inplace=True)

    sub.to_csv("merged_museum_valid_links.csv", index=False)


def cluster():
    sub = pd.read_csv('merged_museum_valid_links.csv')
    words = list(set(sub['classification'].to_list()))
    words.pop(0)

    phrases = []

    for word in words:
        word = re.sub("[()&]", "", word)
        phrases.append(list(filter(None, re.split("[-\s/]", word))))


    g = gensim.downloader.load('word2vec-google-news-300')

    print(g["terracotta"].shape)

    word_vectors = []

    p = inflect.engine()

    for phrase in phrases:
        print("working on", phrase)
        meaning = np.zeros((300,))
        for word in phrase:
            word = p.singular_noun(word.lower()) or word
            if word in g:
                meaning += g[word]
            else:
                print(word)
        
        word_vectors.append(meaning)

    # print(glove_vectors.most_similar(words))



    

    centroids, _ = kmeans(word_vectors, k_or_guess=30)

    word_clusters = np.argmin([
        [spatial.distance.euclidean(wv, cv) for cv in centroids]
        for wv in word_vectors
    ], 1)

    classes = {}

    for k in range(centroids.shape[0]):
        for i, word in enumerate(words):
            if word_clusters[i] == k:
                classes[word] = k
    
    labels = []

    for word in sub['classification']:
        if word in classes:
            labels.append(classes[word])
        else:
            labels.append(30)

    print(labels)

    sub = sub.assign(label = labels)

    sub.to_csv("merged_labels.csv", index=False)




if __name__ == "__main__":
    # cluster()
    # valid_files()
    # DIR2 = 'merged_museum_images'

    # for name in os.listdir(DIR2):
    #     file_path = os.path.join(DIR2, name)

    #     if(os.path.getsize(file_path) > 0):
    #         img = Image.open(file_path)
            
    #         if img.mode == "L":
    #             print('hi', file_path)
    #             fake_rgb = Image.merge("RGB", (img, img, img))
    #             fake_rgb.save(file_path)

    df = pd.read_csv('merged_labels.csv')
    str_cols = df.columns[df.dtypes.eq('object')]
    clfs = {c:LabelEncoder() for c in str_cols}
    for col, clf in clfs.items():
        df[col] = clfs[col].fit_transform(df[col])

    print(df)

    df.to_csv("merged_numerical_labels.csv")

            

    
