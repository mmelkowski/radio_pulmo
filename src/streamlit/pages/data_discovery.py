import streamlit as st
import pickle
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import numpy as np
import plotly.express as px
import pandas as pd
import pathlib

# import custom Navigation bar
from modules.nav import Navbar
Navbar()

titre_text = """
# Exploration des données:
"""
st.markdown(titre_text)

text_3 = """
<div style="text-align: justify;">

## Pre-processing des données

Le preprocessing suivant e sur les données : 

- Import des images et des masques depuis les répertoire liés à la catégorie et au type
- Redimensionnement du masque et de l'image en 256x256
- Application du masque pour isoler l'information concernant les poumons
- Import de l'information sur l'origine de l'image dans les metadata
- Transformation en jeu de données tabulaire

Le jeu de données préprocessé contient 21 165 images après masquage réparties en quatre catégories.


</div>
"""
st.markdown(text_3,unsafe_allow_html=True)

# Ouvrir le dataframe
df_small_path = pathlib.Path('resources/df_small.pkl')
with open(df_small_path, 'rb') as f:
    df_small = pickle.load(f)


#### Sélection de 2 images par type
sampled_images = df_small.groupby('label').apply(lambda x: x.sample(n=2, random_state=42)).reset_index(drop=True)

# Afficher le dataframe en cliquant sur un bouton 
if st.button("Afficher le jeu de données"):
    st.write(sampled_images)

vis_text = """
## Visualisation d'un échantillon d'images 
"""
st.markdown(vis_text)

cols = st.columns(4)

show_centered_images = st.checkbox("Augmenter le contraste et la saturation")

new_images = st.checkbox("Afficher de nouvelles images")
if new_images:
    sampled_images = df_small.groupby('label').apply(lambda x: x.sample(n=2)).reset_index(drop=True)

# Répartir les images sur 4 colonnes
for i, (index, row) in enumerate(sampled_images.iterrows()):
    col_index = i % 4 
    img_array = row['image']
    if show_centered_images:
        img = img_array.reshape(256, 256)
        img = Image.fromarray(img)
        img = img.convert('L')
        # Augmenter le contraste de l'image
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5) 
        # Agmenter la saturation
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.2)
        cols[col_index].image(img, caption=f'{row["nom"]}')
    else:
        img = img_array.reshape(256, 256)
        img = Image.fromarray(img)
        img = img.convert('L')
        cols[col_index].image(img, caption=f'{row["nom"]}')

text_6 = """
<div style="text-align: justify;">

## Statistiques par catégories et par source

Le tableau et les boites de dispersion suivantes résument les différences observées sur les pixels.

</div>
"""

st.markdown(text_6, unsafe_allow_html=True)

# Charger le dataset contenant uniquement label et les statistiques basiques par image
@st.cache_data
def load_data():
    df_mean_std_path = pathlib.Path('resources/df_mean_std.pkl')
    df = pd.read_pickle(df_mean_std_path)
    return df

# Charger les données
df_mean_std = load_data()

# Demander à l'utilisateur sur quoi il veut se baser pour faire le countplot
options = ['Label', 'Source']
selection = st.selectbox("Choisir l'information à utiliser pour les boites de dispersion ", options)

if selection == 'Label':
    result = df_mean_std.groupby('label').agg(Moyenne=('mean_pixel', 'mean'), EcartType=('std_pixel', 'mean'), Minimum=('min_pixel', 'mean'), Maximum=('max_pixel', 'mean'),  Mediane=('median_pixel', 'mean'))
if selection == 'Source':
    result = df_mean_std.groupby('source').agg(Moyenne=('mean_pixel', 'mean'), EcartType=('std_pixel', 'mean'), Minimum=('min_pixel', 'mean'), Maximum=('max_pixel', 'mean'),  Mediane=('median_pixel', 'mean'))

st.write(result)

if selection == 'Label':
    fig = px.box(df_mean_std, x='label', y='mean_pixel', color='label', title="Distribution de la valeur moyenne des pixels par image et par Label", labels={'label': 'Label'})
if selection == 'Source':
    fig = px.box(df_mean_std, x='source', y='mean_pixel', color='source', title="Distribution de la valeur moyenne des pixel par image et par Source", labels={'source': 'Source'})


# Afficher le countplot dans Streamlit
st.plotly_chart(fig)

text_7 = """
<div style="text-align: justify;">

Il semble que la moyenne des données COVID soient plus élevée, ce qui suggère des images plus blanches et donc une cohérence avec la plus grande opacité attendue.

Une autre possibilité de visualisation est de tracer le nuage de point de la valeur moyenne du pixel en ordonnées et les pixels en abscisse et la moyenne en fonction de l’écart-type.

</div>
"""
st.markdown(text_7, unsafe_allow_html=True)

fig1 = px.scatter(
        df_mean_std, x='mean_pixel', y='std_pixel', color='label', title="Moyenne et Ecart-Type des Pixels par Label ", 
        labels={'mean_pixel': 'Moyenne des Pixels', 'std_pixel': 'Ecart-Type des Pixels'},
        #template="plotly_dark"
        )  # Utilisation d'un fond sombre
fig1.update_traces(marker=dict(size=2))

# Afficher le dataframe avec un bouton 
if st.button("Afficher le graphique de l'écart type en fonction de la moyenne"):
    # Si le bouton est cliqué, afficher le jeu de données
    st.plotly_chart(fig1)


# Créer le graphique avec Plotly Express
def create_plot(label):
    filtered_df = df_mean_std[df_mean_std['label'] == label]
    
    fig = px.scatter(
        filtered_df, x='mean_pixel', y='std_pixel', color='label',
        title=f"Moyenne et Ecart-Type des Pixels pour le label {label}", 
        labels={'mean_pixel': 'Moyenne des Pixels', 'std_pixel': 'Ecart-Type des Pixels'}
    )
    fig.update_traces(marker=dict(size=3))
    return fig

# Afficher le dataframe avec un bouton 
label = st.selectbox("Afficher le graphique de l'écart-type en fonction de la moyenne pour une seule catégorie pour plus de lisibilité ?", 
                     options=["Aucune sélection"] + list(df_mean_std['label'].unique()), 
                     index=0)  # "Aucune sélection" sera la première option

# Vérifier si l'utilisateur a sélectionné une catégorie autre que "Aucune sélection"
if label != "Aucune sélection":
    # Créer le graphique basé sur le label sélectionné
    fig2 = create_plot(label)
    
    # Afficher le graphique
    st.plotly_chart(fig2)

#
text_8 = """
<div style="text-align: justify;">

Dans les deux représentations, on remarque que les valeurs des pixels diffèrent en fonction des catégories et des sources de données. 
Les données semblent plus dispersées pour les catégories COVID et Lung Opacity.
Les valeurs sont en moyenne plus élevées pour les catégories COVID et Viral Pneumonia.

## Images moyennes par type et par source

Pour aller plus loin, on peut tracer l’image moyenne par groupe en fonction de la source de données.
</div>
"""
st.markdown(text_8, unsafe_allow_html=True)

# Ouvrir le dataframe
df_avg_img_path = pathlib.Path('resources/df_avg_img.pkl')
with open(df_avg_img_path, 'rb') as g:
    df_avg_img = pickle.load(g)


cols2 = st.columns(5)

    # Répartir les images sur 5 colonnes
for i, (index, row) in enumerate(df_avg_img.iterrows()):
    col_index = i % 5 

    img_array = row['image']

    img = img_array.reshape(256, 256)
    img = Image.fromarray(img)
    img = img.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5) 
        # Agmenter la saturation
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.2)
    # Pour résoudre le problème d'alignement on diminue la longueur
    short_caption = row["label_source"][:17]
    cols2[col_index].image(img, caption=f'{short_caption}')
    #cols2[col_index].image(img, caption=f'{[row["label_source"]]}')

text_9 = """
<div style="text-align: justify;">

### Interprétation

Il semble y avoir un biais sur les radiographie pulmonaires liées à la pneumonie virale : le cœur est proportionnellement plus important et l’application du masque efface la partie inférieure du poumon gauche. 
En revanche, les données étiquetées comme normale de la source "pneumonia-chestxray” ne semble pas présenter cette déformation.

On émet l’hypothèse que les données associés à la pneumonie sont des radiographies d’enfants.

Un biais risque donc d'être présent dans la modélisation de la catégorie Viral Pneumonia.
"""
st.markdown(text_9, unsafe_allow_html=True)