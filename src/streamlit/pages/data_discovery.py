import streamlit as st
import pickle
import matplotlib.pyplot as plt
from PIL import Image
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

# intro_text = """
# # Prise en main et découverte des données:
# ## Visualisation de quelques images du dataset:
# """
# st.markdown(intro_text)
# st.image("resources/decouverte_donnees/showcase.png")

# text_2 = """
# <div style="text-align: justify;">

# Les images sont des radiographies pulmonaires provenant de différentes sources, on remarque rapidement que seule une partie de l'image nous intéressera puisqu'on s'intéressera aux poumons alors que l'image laisse notamment la colonne vertébrale et le cœur. Il faudra donc appliquer les masques qui ont été précalculés.

# Les données sont en échelle de gris, une importation en couleur donne trois valeurs identiques pour les différents canaux. Chaque image correspond donc à une matrice de 299*299. Ci-dessous, un exemple d’image sur les trois canaux RGB.
# </div>"""
# st.markdown(text_2,unsafe_allow_html=True)
# st.image("resources/decouverte_donnees/x-ray_rgb.png")

text_3 = """
<div style="text-align: justify;">

## Pre-processing des données

Le preprocessing suivant e sur les données : 

- Import des images et des masques depuis les répertoire liés à la catégorie et au type
- Redimensionnement du masque et de l'image en 256x256
- Application du masque pour isoler l'information concernant les poumons
- Import de l'information sur l'origine de l'image dans les metadata
- Transformation en jeu de données tabulaire

Le jeu de données préprocessé utilisé pour notre étude contient 21 165 images après masquage réparties en quatre classes.
Un sous-échantillon du jeu de données préprocessé est présenté ci-dessous.

</div>
"""
st.markdown(text_3,unsafe_allow_html=True)

# Ouvrir le dataframe
df_small_path = pathlib.Path('resources/df_small.pkl')
with open(df_small_path, 'rb') as f:
    df_small = pickle.load(f)


#### A ameliorer pour afficher 2 images

sampled_images = df_small.sample(n=8) 

# Afficher le DataFrame 
st.write(sampled_images)


vis_text = """
## Visualisation d'un échantillon d'images 
"""
st.markdown(vis_text)
#st.image("resources/decouverte_donnees/showcase.png")


# Sélectionner 8 images au hasard

cols = st.columns(4)

#### Si possible rendre fonctionnel l affichage apres centrage

show_centered_images = st.checkbox("Afficher les images après centrage")

# Répartir les images sur 3 colonnes, 2 images par colonne
for i, (index, row) in enumerate(sampled_images.iterrows()):
    col_index = i % 4 

    img_array = row['image']

    if show_centered_images:
        img_array = (img_array-127)/127
        #img_array = np.clip(img_array, -1, 1)
        #img = Image.fromarray(((img_array + 1) * 127).astype(np.uint8))
        img = Image.fromarray(img)
        img = img.convert('RGB')
        cols[col_index].image(img, caption=f'{row["nom"]}')
    else:
        img = img_array.reshape(256, 256)
        img = Image.fromarray(img)
        img = img.convert('L')
        cols[col_index].image(img, caption=f'{row["nom"]}')

#st.image("resources/decouverte_donnees/table.png")

# text_4 = """
# <div style="text-align: justify;">

# ## Statistiques descriptives sur les données
# ### Répartition des données par catégorie

# La première analyse réalisée est une visualisation du nombre d'images par catégorie. 
# </div>
# """
# st.markdown(text_4, unsafe_allow_html=True)
# st.image("resources/decouverte_donnees/barplot_disease.png")

# text_5 = """
# <div style="text-align: justify;">

# On observe que nous n’avons majoritairement des données de radio pulmonaires pour les patients ne présentant pas d’affection. Les données de pneumonie virales sont quant à elles minoritaires.

# ### Répartition des données par source et par catégorie.

# Les images proviennent de plusieurs sources différentes. On cherche à visualiser la répartition des catégories par groupe afin d’identifier un éventuel biais.

# </div>
# """
# st.markdown(text_5, unsafe_allow_html=True)
# st.image("resources/decouverte_donnees/data_origin.png")

text_6 = """
<div style="text-align: justify;">

## Statistiques par catégories et par source

Le tableau et les boites de dispersion suivantes résument les différences observées sur les pixels.

</div>
"""

st.markdown(text_6, unsafe_allow_html=True)
#st.image("resources/decouverte_donnees/boxplot.png")

# Charger le dataset contenant uniquement label, source, moyenne et std par image
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


##### A ameliorer pour afficher aussi min et max mais il faut reprendre le dataset
if selection == 'Label':
    result = df_mean_std.groupby('label').agg(Moyenne=('mean_pixel', 'mean'), EcartType=('std_pixel', 'mean'), Minimum=('min_pixel', 'mean'), Maximum=('max_pixel', 'mean'),  Mediane=('median_pixel', 'mean'))
if selection == 'Source':
    result = df_mean_std.groupby('source').agg(Moyenne=('mean_pixel', 'mean'), EcartType=('std_pixel', 'mean'), Minimum=('min_pixel', 'mean'), Maximum=('max_pixel', 'mean'),  Mediane=('median_pixel', 'mean'))

st.write(result)

if selection == 'Label':
    fig = px.box(df_mean_std, x='label', y='mean_pixel', color='label', title="Distribution de la valeur moyenne des pixels par image et par Label", labels={'label': 'Label'})
if selection == 'Source':
    fig = px.histogram(df_mean_std, x='source', y='mean_pixel', color='source', title="Distribution de la valeur moyenne des pixel par image et par Source", labels={'source': 'Source'})


# Afficher le countplot dans Streamlit
st.plotly_chart(fig)


text_7 = """
<div style="text-align: justify;">

Il semble que la moyenne des données COVID soient plus élevée, ce qui suggère des images plus blanches et donc une cohérence avec la plus grande opacité attendue.

Une autre possibilité de visualisation est de tracer le nuage de point de la valeur moyenne du pixel en ordonnées et les pixels en abscisse et la moyenne en fonction de l’écart-type.

</div>
"""
st.markdown(text_7, unsafe_allow_html=True)

fig = px.scatter(
        df_mean_std, x='mean_pixel', y='std_pixel', color='label', title="Moyenne et Ecart-Type des Pixels par Label ", 
        labels={'mean_pixel': 'Moyenne des Pixels', 'std_pixel': 'Ecart-Type des Pixels'},
        #template="plotly_dark"
        )  # Utilisation d'un fond sombre

fig.update_traces(marker=dict(size=2))

st.plotly_chart(fig)

### A améliorer pour demander si on veut plus d'info et idealement conserver les couleurs entre les deux graphs.

text_8 = """
<div style="text-align: justify;">
Le graphique ci-dessous permet de visualiser les catégories une par une pour améliorer la lisibilté
</div>
"""
st.markdown(text_8, unsafe_allow_html=True)


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

# Récupérer tous les labels uniques
labels = df_mean_std['label'].unique()

# Initialiser l'index du label actuel dans la session
if 'label_index' not in st.session_state:
    st.session_state.label_index = 0  # Commencer avec le premier label

# Afficher le graphique pour le label actuel
current_label = labels[st.session_state.label_index]
fig = create_plot(current_label)

# Afficher le graphique Plotly
st.plotly_chart(fig)

# Boutons pour naviguer entre les labels
col1, col2 = st.columns(2)
with col1:
    if st.button('Précédent'):
        # Mettre à jour l'index pour revenir au label précédent
        st.session_state.label_index = (st.session_state.label_index - 1) % len(labels)

with col2:
    if st.button('Suivant'):
        # Mettre à jour l'index pour passer au label suivant
        st.session_state.label_index = (st.session_state.label_index + 1) % len(labels)


# st.image("resources/decouverte_donnees/relplot.png")

# st.markdown("""<div style="text-align: justify;">On visualise également l’écart-type en fonction de la moyenne pour chacune des catégories.</div>""", unsafe_allow_html=True)
# st.image("resources/decouverte_donnees/dispersion.png")

text_8 = """
<div style="text-align: justify;">

Dans les deux représentations, on remarque que les valeurs des pixels diffèrent en fonction des catégories. Les données semblent également plus dispersées pour les catégories COVID et Lung Opacity.
Les valeurs sont plus extrêmes dans le cas de la catégorie "Viral Pneumonia".

## Images moyennes par type et par source

Pour aller plus loin, on trace l’image moyenne par groupe.
</div>
"""
st.markdown(text_8, unsafe_allow_html=True)

# Ouvrir le dataframe
df_avg_img_path = pathlib.Path('resources/df_avg_img.pkl')
with open(df_avg_img_path, 'rb') as g:
    df_avg_img = pickle.load(g)


cols2 = st.columns(5)

# Ameliorer la visuatlisation des images et la mise en page

# Répartir les images sur 3 colonnes, 2 images par colonne
for i, (index, row) in enumerate(df_avg_img.iterrows()):
    col_index = i % 5 

    img_array = row['image']

    img = img_array.reshape(256, 256)
    img = Image.fromarray(img)
    img = img.convert('RGB')
    cols2[col_index].image(img, caption=f'{row["label_source"]}')


#st.image("resources/decouverte_donnees/mean.png")

text_9 = """
<div style="text-align: justify;">

### Interprétation

Il semble y avoir un biais sur les analyses radio pulmonaires liées à la pneumonie virale : le cœur est proportionnellement plus important et l’application du masque efface la partie inférieure du poumon gauche. 
Les données de la source “pneumonia-chestxray” ne semble pas présenter cette déformation.

Sans certitude, on émet l’hypothèse que les données associés à la pneumonie virale sont des radiographies pulmonaires d’enfants.
Un biais risque donc d'être présent dans la modélisation de cette catégorie.
"""
st.markdown(text_9, unsafe_allow_html=True)