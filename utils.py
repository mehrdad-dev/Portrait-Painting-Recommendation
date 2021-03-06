from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import random


IMAGE_SIZE = (224, 224)
IMAGES_PATH = '/images/'


# =============================================================================
def Images2Matrix(images):
    images = list()
    for image in tqdm(range(len(images) - 1)):
        tmp = img_to_array(
            load_img(IMAGES_PATH + images[image],
            target_size=IMAGE_SIZE)
        )

        expand = np.expand_dims(tmp, axis=0)
        images.append(expand)

    return preprocess_input(np.vstack(images))


def CosSim(model, images_matrix, images_list):
    features_extractor= Model(
        inputs=model.input, outputs=model.layers[-2].output)
    imgs_features=features_extractor.predict(images_matrix)
    cosSimilarities=cosine_similarity(imgs_features)
    cos_similarities_df=pd.DataFrame(cosSimilarities,
                                        columns=images_list[:len(
                                            images_list) - 1],
                                        index=images_list[:len(images_list) - 1])
    return cos_similarities_df


def Recommend(cossim_table, input_img, k_recommend):

    original=load_img(input_img, target_size=(224, 224))
    img_name=input_img.split('/')[-1]

    closest_imgs=cossim_table[img_name].sort_values(ascending=False)[
                                                     1:k_recommend+1].index
    closest_imgs_scores=cossim_table[img_name].sort_values(ascending=False)[
                                                            1:k_recommend+1]

    return closest_imgs, closest_imgs_scores
