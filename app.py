import streamlit as st
import numpy as np
import pandas as pd
import os
import random
import utils
import tensorflow as tf

images = ['images/'+img for img in os.listdir("images/")]
cossim_path = 'cosine_similarity/cos_sim.pkl'
model_path = 'model/'
# =============================================================================

st.title('PPR - Portrait Painting Recommendation')
st.markdown('A recommendation system for images (specifically painted portraits)',
            unsafe_allow_html=True)
star = " 🌟 If you like the idea give it a star on the [GitHub]()"
st.markdown(star, unsafe_allow_html=True)

expander1 = st.expander("📊 Data")
data_ex = """
**Dataset page:** [Wikiart](https://data.mendeley.com/datasets/289kxpnp57/1).

**Published:** 14 Jan 2021.

**DOI:** 10.17632/289kxpnp57.1
"""
expander1.markdown(data_ex)

expander2 = st.expander("🤖 Model ")
model_ex = """
**Feature Extractor:** MobileNetv1

**Similarity Metric:** Cosine Similarity
"""
expander2.markdown(model_ex)


# =============================================================================


def gen_random_images(images, k=3):
    random_images = random.sample(images, k)
    return random_images

random_images = gen_random_images(images, k=3)
st.image(random_images, width=224, use_column_width=False,
            caption=['image 1', 'image 2', 'image 3'])

# =============================================================================

st.markdown('<h4>Select a random image to get recommendation</h4>', unsafe_allow_html=True)
selected_image = st.selectbox('', ('image 1', 'image 2', 'image 3'))

if selected_image == 'image 1':
    selected_image = random_images[0]
elif selected_image == 'image 2':
    selected_image = random_images[1]
else:
    selected_image = random_images[2]

st.image(selected_image, width=224,
         use_column_width=False, caption='Your Selected Image')

# =============================================================================


@st.cache
def expensive_compute(images):
    images_matrix = utils.Images2Matrix(images)
    cos_sim = utils.CosSim(model, images_matrix, images)
    return images_matrix, cos_sim


@st.cache
def pre_recommend(cossim_path, model_path):
    model = tf.keras.models.load_model(model_path)
    cos_sim = pd.read_pickle(cossim_path)

    return model, cos_sim


model, cos_sim = pre_recommend(cossim_path, model_path)

recom_reuslts, recom_scores = utils.Recommend(
    cos_sim, selected_image, k_recommend=6)

# =============================================================================

st.markdown('<h4>Recomendation Results: /h4>', unsafe_allow_html=True)
print('recom_reuslts', recom_reuslts)
print('recom_scores', recom_scores)

print(recom_scores.values())
print(recom_scores.keys())


st.image(recom_reuslts[:3], width=224, use_column_width=False,
         caption=[f'Similarity Score: {recom_scores[recom_reuslts[0]]}',
                  f'Similarity Score: {recom_scores[recom_reuslts[1]]}',
                  f'Similarity Score: {recom_scores[recom_reuslts[2]]}']
         )

st.image(recom_reuslts[4:], width=224, use_column_width=False,
         caption=[f'Similarity Score: {recom_scores[3]}',
                  f'Similarity Score: {recom_scores[4]}',
                  f'Similarity Score: {recom_scores[5]}']
         )
