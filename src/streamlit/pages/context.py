import streamlit as st

# import custom Navigation bar
from modules.nav import Navbar
Navbar()


st.title("Contexte du projet")
context_text = """
<div style="text-align: justify;">

Une équipe de chercheurs de multiples universités du Moyen Orient et Asie ont assemblé un jeu de données de radiographie du thorax pour des patients, sain, atteint du  Covid, de pneumonie et d'opacité pulmonaire.

Ce [dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) a été constitué dans le but d'utiliser le deep learning pour le diagnostic du Covid-19 par X-ray plutôt que par RT-PCR.

Il ya 4 sources d'images présent dans le dataset (avec entre parenthèse le nombre d'images):

| Source                 | Nombre d'images |
| ---------------------- | --------------- |
| COVID-19 data          | 3615            |
| Normal images          | 10192           |
| Lung opacity images    | 6012            |
| Viral Pneumonia images | 1345            |

Chaque image vient avec son masque pré-calculé, généré par apprentissage semi-automatique.

Par son application ont peut donc isoler les poumons et avoir des données concernant uniquement les poumons.
</div>"""
st.markdown(context_text, unsafe_allow_html=True)
st.image("resources/intro/mask_process.png")

objective_text = """
<div style="text-align: justify;">

## Objectif

On cherche à utiliser ce jeu de données pour aider au diagnostic d'affection pulmonaire. Ce sujet se présente comme un problème de classification à plusieurs classes. 

On s'attachera ici à minimiser le nombre de faux-négatifs i.e des patients pour lesquels on considérerait que la radiographie est normale alors qu'il est sujet à une infection.
</div>"""
st.markdown(objective_text, unsafe_allow_html=True)
