import streamlit as st
import pickle
from PIL import Image, ImageEnhance
import plotly.express as px
import pandas as pd
import pathlib

# import custom Navigation bar
from modules.nav import Navbar

Navbar()

#config:
path_to_resources = pathlib.Path("src/streamlit/resources")

titre_text = """
# Exploration des données:
"""
st.markdown(titre_text)

text_3 = """
<div style="text-align: justify;">

## Pre-processing des données

Le preprocessing suivante sur les données : 

- Import des images et des masques depuis les répertoire liés à la catégorie et au type
- Redimensionnement du masque et de l'image en 256x256
- Application du masque pour isoler l'information concernant les poumons
- Import de l'information sur l'origine de l'image dans les metadata
- Transformation en jeu de données tabulaire

Le jeu de données préprocessé contient 21 165 images après masquage réparties en quatre catégories.


</div>
"""
st.markdown(text_3, unsafe_allow_html=True)

# Ouvrir le dataframe
df_small_path = path_to_resources / "decouverte_donnees" / "df_small.pkl"
with open(df_small_path, "rb") as f:
    df_small = pickle.load(f)


#### Sélection de 2 images par type
sampled_images = (
    df_small.groupby("label")
    .apply(lambda x: x.sample(n=2, random_state=42))
    .reset_index(drop=True)
)

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
    sampled_images = (
        df_small.groupby("label").apply(lambda x: x.sample(n=2)).reset_index(drop=True)
    )

# Répartir les images sur 4 colonnes
for i, (index, row) in enumerate(sampled_images.iterrows()):
    col_index = i % 4
    img_array = row["image"]
    if show_centered_images:
        img = img_array.reshape(256, 256)
        img = Image.fromarray(img)
        img = img.convert("L")
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
        img = img.convert("L")
        cols[col_index].image(img, caption=f'{row["nom"]}')

text_6 = """
<div style="text-align: justify;">

## Statistiques par catégories et par source

Le tableau et les boites de dispersion suivantes résument les différences observées sur les pixels.

</div>
"""

st.markdown(text_6, unsafe_allow_html=True)


# Charger le dataset contenant uniquement label et les statistiques basiques par image
#@st.cache_data
def load_data():
    df_mean_std_path = path_to_resources / "decouverte_donnees" / "df_mean_std.pkl"
    df = pd.read_pickle(df_mean_std_path)
    return df


# Charger les données
df_mean_std = load_data()

# Demander quel info utiliser pour faire le countplot
options = ["Label", "Source"]
selection = st.selectbox(
    "Choisir l'information à utiliser pour les boites de dispersion ", options
)

if selection == "Label":
    result = df_mean_std.groupby("label").agg(
        Moyenne=("mean_pixel", "mean"),
        EcartType=("std_pixel", "mean"),
        Minimum=("min_pixel", "mean"),
        Maximum=("max_pixel", "mean"),
        Mediane=("median_pixel", "mean"),
    )
if selection == "Source":
    result = df_mean_std.groupby("source").agg(
        Moyenne=("mean_pixel", "mean"),
        EcartType=("std_pixel", "mean"),
        Minimum=("min_pixel", "mean"),
        Maximum=("max_pixel", "mean"),
        Mediane=("median_pixel", "mean"),
    )

st.write(result)

if selection == "Label":
    fig = px.box(
        df_mean_std,
        x="label",
        y="mean_pixel",
        color="label",
        title="Distribution de la valeur moyenne des pixels par image et par Label",
        labels={"label": "Label"},
    )
if selection == "Source":
    fig = px.box(
        df_mean_std,
        x="source",
        y="mean_pixel",
        color="source",
        title="Distribution de la valeur moyenne des pixel par image et par Source",
        labels={"source": "Source"},
    )


# Afficher le countplot dans Streamlit
st.plotly_chart(fig)

text_7 = """
<div style="text-align: justify;">

Il semble que la moyenne des données COVID soient plus élevée, ce qui suggère des images plus blanches et donc une cohérence avec la plus grande opacité attendue.

Une autre possibilité de visualisation est de tracer le nuage de point de la valeur moyenne du pixel en ordonnées et les pixels en abscisse et la moyenne en fonction de l’écart-type.

</div>
"""
st.markdown(text_7, unsafe_allow_html=True)


# Créer le graphique avec Plotly Express
def create_plot():
    label_choice = ["Toutes les catégories"] + df_mean_std["label"].unique().tolist()
    label = st.selectbox("Pour toutes les catégories ou une spécifique ?", label_choice)

    if label == "Toutes les catégories":
        filtered_df = df_mean_std
    else:
        filtered_df = df_mean_std[df_mean_std["label"] == label]

    fig = px.scatter(
        filtered_df,
        x="mean_pixel",
        y="std_pixel",
        color="label",
        title=f"Moyenne et Ecart-Type des Pixels pour le label {label}",
        labels={
            "mean_pixel": "Moyenne des Pixels",
            "std_pixel": "Ecart-Type des Pixels",
        },
    )
    fig.update_traces(marker=dict(size=3))
    return fig


# Afficher le dataframe avec un bouton
label = st.selectbox(
    "Afficher le graphique de l'écart-type en fonction de la moyenne ?",
    options=["Non"] + ["Oui"],
    index=0,
)

# Vérifier si l'utilisateur a sélectionné une catégorie autre que "Aucune sélection"
if label == "Oui":
    # Créer le graphique basé sur le label sélectionné
    fig2 = create_plot()

    # Afficher le graphique
    st.plotly_chart(fig2)


# Ouvrir le dataframe des images moyennes.
df_avg_img_path = path_to_resources / "decouverte_donnees" / "df_avg_img.pkl"
with open(df_avg_img_path, "rb") as g:
    df_avg_img = pickle.load(g)


def afficher_pixels_par_categorie(df):

    # Aplatir les images en un tableau 1D
    def flatten_image(image):
        return [pixel for row in image for pixel in row]

    # Créer nouvelle colonne
    df["flattened_image"] = df["image"].apply(flatten_image)

    # Convertir la colonne flattened_image en DataFrame
    flat_images = pd.DataFrame(df["flattened_image"].tolist(), index=df["label_source"])

    # Transformer les données pour plotly
    flat_images = flat_images.stack().reset_index()
    flat_images.columns = ["label_source", "pixel_index", "pixel_value"]

    # Sélection de la catégorie
    category_choice = ["Toutes les catégories"] + df["label_source"].unique().tolist()
    category = st.selectbox(
        "Pour toutes les catégories ou une spécifique ?", category_choice
    )

    # Filtrer les données pour la catégorie sélectionnée
    if category == "Toutes les catégories":
        filtered_data = flat_images
    else:
        filtered_data = flat_images[flat_images["label_source"] == category]

    # Créer un scatter plot avec plotly
    fig3 = px.scatter(
        filtered_data,
        x="pixel_index",
        y="pixel_value",
        color="label_source",
        title=f"Valeurs des pixels pour {category}",
        labels={"pixel_index": "Index des Pixels", "pixel_value": "Valeur des Pixels"},
    )
    fig3.update_traces(marker=dict(size=1))
    # Afficher le graphique
    st.plotly_chart(fig3)


# Afficher le dataframe avec un bouton
label_mean = st.selectbox(
    "Afficher le graphique de l'écart-type de la valeur moyenne des pixels pour une catégorie x source ?",
    options=["Non"] + ["Oui"],
    index=0,
)  # "Aucune sélection" sera la première option
if label_mean == "Oui":
    afficher_pixels_par_categorie(df_avg_img)


text_8 = """
<div style="text-align: justify;">

On remarque que les valeurs des pixels et graphiques associés diffèrent en fonction des catégories et des sources de données. 
Les données semblent plus dispersées pour les catégories COVID et Lung Opacity.
Les valeurs sont en moyenne plus élevées pour les catégories COVID et Viral Pneumonia.

## Images moyennes par type et par source

Pour aller plus loin, on peut tracer l’image moyenne par groupe en fonction de la source de données.
</div>
"""
st.markdown(text_8, unsafe_allow_html=True)


cols2 = st.columns(5)

# Répartir les images sur 5 colonnes
for i, (index, row) in enumerate(df_avg_img.iterrows()):
    col_index = i % 5

    img_array = row["image"]

    img = img_array.reshape(256, 256)
    img = Image.fromarray(img)
    img = img.convert("L")
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    # Agmenter la saturation
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.2)
    # Pour résoudre le problème d'alignement on diminue la longueur
    short_caption = row["label_source"][:17]
    cols2[col_index].image(img, caption=f"{short_caption}")
    # cols2[col_index].image(img, caption=f'{[row["label_source"]]}')

text_9 = """
<div style="text-align: justify;">

### Interprétation

Il semble y avoir un biais sur les radiographie pulmonaires liées à la pneumonie virale : le cœur est proportionnellement plus important et l’application du masque efface la partie inférieure du poumon gauche. 
En revanche, les données étiquetées comme normale de la source "pneumonia-chestxray” ne semble pas présenter cette déformation.

On émet l’hypothèse que les données associés à la pneumonie sont des radiographies d’enfants.

Un biais risque donc d'être présent dans la modélisation de la catégorie Viral Pneumonia.
"""
st.markdown(text_9, unsafe_allow_html=True)


# Crédit
bottom_text = """
<div style="font-size: 14px; color: gray; font-style: italic; text-align: center; margin-top: 20px;">
 Cette application a été développée par 
    <br>
    <a href="https://www.linkedin.com/in/chris-hozé-007901a5" target="_blank" style="color: #0073e6;">Chris Hozé</a> 
    et 
    <a href="https://www.linkedin.com/in/mickael-melkowski/" target="_blank" style="color: #0073e6;">Mickaël Melkowski</a>.

</div>
"""

# Affichage du texte dans la sidebar
st.sidebar.markdown(bottom_text, unsafe_allow_html=True)