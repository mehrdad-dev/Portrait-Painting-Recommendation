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
star = " 🌟 If you like the idea give it a star on the [GitHub](https://github.com/mehrdad-dev/Portrait-Painting-Recommendation)"
st.markdown(star, unsafe_allow_html=True)

desc = " 📖 Full Description of the project: [PDF](https://github.com/mehrdad-dev/Portrait-Painting-Recommendation/blob/main/assets/project1.pdf)"
st.markdown(desc, unsafe_allow_html=True)

expander1 = st.expander("📊 Data")
data_ex = """
**Dataset page:** [Wikiart](https://data.mendeley.com/datasets/289kxpnp57/1).

**Dataset size:** 926 sample in 6 artistic style.

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

st.markdown('<h4>Your selected image</h4>', unsafe_allow_html=True)
st.image(selected_image, width=224,
         use_column_width=False, caption='Selected Image')

star = "If you got an error, just choose another image from the options in the drop-down section."
st.markdown(star, unsafe_allow_html=True)

# =============================================================================


@st.cache
def expensive_compute(images):
    images_matrix = utils.Images2Matrix(images)
    cos_sim = utils.CosSim(model, images_matrix, images)
    return images_matrix, cos_sim


@st.cache(allow_output_mutation=True)
def pre_recommend(cossim_path, model_path):
    model = tf.keras.models.load_model(model_path)
    cos_sim = pd.read_pickle(cossim_path)

    return model, cos_sim


model, cos_sim = pre_recommend(cossim_path, model_path)

recom_reuslts, recom_scores = utils.Recommend(
    cos_sim, selected_image, k_recommend=6)

# =============================================================================

st.markdown('<h4>Recomendation Results: </h4>', unsafe_allow_html=True)
recom_scores = recom_scores.tolist()
recom_reuslts = recom_reuslts.tolist()
recom_reuslts_fixed = []
for result in recom_reuslts:
    recom_reuslts_fixed.append('images/'+result)

st.image(recom_reuslts_fixed[:3], width=224, use_column_width=False,
         caption=['Similarity Score: {:.3f}'.format(recom_scores[0]),
                  'Similarity Score: {:.3f}'.format(recom_scores[1]),
                  'Similarity Score: {:.3f}'.format(recom_scores[2])]
         )

st.image(recom_reuslts_fixed[3:], width=224, use_column_width=False,
         caption=['Similarity Score: {:.3f}'.format(recom_scores[3]),
                  'Similarity Score: {:.3f}'.format(recom_scores[4]),
                  'Similarity Score: {:.3f}'.format(recom_scores[5])]
         )
