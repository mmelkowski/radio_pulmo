import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px

# import custom Navigation bar
from modules.nav import Navbar
Navbar()

st.title("Contexte du projet")


avant_propos_text = """
<div style="text-align: justify;">

## Avant Propos : 
Ce travail a été réalisé par [Chris Hozé](https://www.linkedin.com/in/chris-hozé-007901a5) et [Mickaël Melkowski](https://www.linkedin.com/in/mickael-melkowski/) dans le cadre de notre formation DataScientist réalisée de Septembre à Décembre 2024 avec DataScientest.

L'objectif de notre étude est de développer un modèle pour classifier les radiographies pulmonaires.

L'ensemble du code, les notebooks d'exploration et les modèles sont disponibles sur le dépot [github](https://github.com/mmelkowski/radio_pulmo/) 

</div>
"""
st.markdown(avant_propos_text, unsafe_allow_html=True)


context_text = """
<div style="text-align: justify;">

## Jeu de données :

Une équipe de chercheurs d'universités du Moyen Orient et Asie ont assemblé un jeu de données de radiographie du thorax pour des patients, sain, atteint du  Covid, de pneumonie et d'opacité pulmonaire.

Ce [dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) a été constitué dans le but de développer un diagnostic automatisé du Covid-19 par radiographie pulmonaire plutôt que par RT-PCR.
Les seules métadonnées disponibles concernent l’origine des radiographies. Aucune information n’est disponible sur les patients hormis le type d'affection.

Chaque image est fournie avec un masque pré-calculé, généré par apprentissage semi-automatique. L'application du masque permet d'isoler les pixels liés aux poumons et ainsi de réduire la zone ciblée, concentrant l'analyse sur la partie pertinente de l'image.

Un exemple d'image, de masque et de radiographie après masquage est présenté ci-dessous :
</div>"""

st.markdown(context_text, unsafe_allow_html=True)

st.image("resources/intro/mask_process.png")

source_tooltip = """Les sources des images ont été recodées de la façon suivante:

    https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data : rnsa

    https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia : pneumonia-chestxray

    https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/#1590858128006-9e640421-6711  : bimcv

    https://github.com/armiro/COVID-CXNet : CXNet

    https://eurorad.org : eurorad          

    https://github.com/ml-workgroup/covid-19-image-repository/tree/master/png  : ml-workgroup

    https://github.com/ieee8023/covid-chestxray-dataset : covid-chestxray

    https://sirm.org/category/senza-categoria/covid-19/ : senza
"""


count_text = """
<div style="text-align: justify;">

4 catégories d'images sont présentes dans le jeu de données, le graphique ci-dessous indique la répartition des images par catégorie et par source :

</div>"""
st.markdown(count_text, unsafe_allow_html=True, help=source_tooltip)


# Charger le dataset contenant uniquement label, source, moyenne et std par image
@st.cache_data
def load_data():
    df = pd.read_pickle('resources/df_mean_std.pkl')
    return df

# Charger les données
df_mean_std = load_data()


# Demander à l'utilisateur sur quoi il veut se baser pour faire le countplot
options = ['Label', 'Source', 'Label & Source']
selection = st.selectbox("Choisir l'information à utiliser pour la répartition ", options)


# Créer le countplot avec Plotly Express
if selection == 'Label':
    fig = px.histogram(df_mean_std, x='label', color='label', title="Nombre d'images par Label", labels={'label': 'Label'})
elif selection == 'Source':
    fig = px.histogram(df_mean_std, x='source', color='label', title="Nombre d'images par Source", labels={'source': 'Source'})
else:  # 'Label & Source'
    fig = px.histogram(df_mean_std, x='label', color='source', title="Nombre d'images par Label et Source",
                        labels={'label': 'Label', 'source': 'Source'})

# Afficher le countplot dans Streamlit
st.plotly_chart(fig)
