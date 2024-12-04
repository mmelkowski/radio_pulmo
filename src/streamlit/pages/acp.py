import streamlit as st
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import plotly.express as px
import pandas as pd

# import custom Navigation bar
from modules.nav import Navbar
Navbar()



titre_text = """
# Analyse en composante principale sur les données:

Une ACP a été réalisé afin de poursuivre l'exploration des données pour identifier un éventuel biais, mais également réaliser un premier essai de classification pour déterminer le type de modèle à utiliser.

"""
st.markdown(titre_text)

text_9 = """
## Analyses en composantes principales

### Part de variance expliquée par les composantes de l’ACP

Pour des raisons de mémoire, l’analyse en composante principale a été réalisée en réduisant les images en dimension 100*100.
La première figure donne le pourcentage de variance sur les 20 premières composantes. On remarque que les dix premières composantes expliquent 51,6 % de la variabilité. Pour atteindre 95% de la variance initiale, il faut conserver 782 composantes.

</div>
"""
st.markdown(text_9, unsafe_allow_html=True)
st.image("resources/decouverte_donnees/ACP.png")

text_10 = """
<div style="text-align: justify;">

### Visualisation sur des quatres premières composantes de l’ACP
</div>
"""
st.markdown(text_10, unsafe_allow_html=True)

PCA_choice = st.radio(
    "**Choix des composantes de l'ACP:**",
    ["PCA1 - PCA2", "PCA3 - PCA4"]
)

if PCA_choice == "PCA1 - PCA2":
    st.markdown("""<div style="text-align: justify;">On visualise les coordonnées des échantillons sur les plans 1 et 2 pour voir s'il est possible de séparer les catégories selon un plan. On étudie également la répartition par source de données pour identifier une éventuelle source de biais. :</div>""", unsafe_allow_html=True)
    st.image("resources/decouverte_donnees/ACP/ACP_1.png")
    st.image("resources/decouverte_donnees/ACP/ACP_2.png")
else:
    st.markdown("""<div style="text-align: justify;">On visualise les coordonnées des échantillons sur les plans 3 et 4 pour voir s'il est possible de séparer les catégories selon un plan. On étudie également la répartition par source de données pour identifier une éventuelle source de biais. :</div>""", unsafe_allow_html=True)
    st.image("resources/decouverte_donnees/ACP/ACP_3.png")
    st.image("resources/decouverte_donnees/ACP/ACP_4.png")

text_11 = """
<div style="text-align: justify;">

### Interprétation

Les axes ne permettent pas de séparer les échantillons par groupe, et on identifie pas de sous-population en fonction de la source de données. 
L’orientation, le niveau de zoom et la taille des poumons diffèrent selon les patients et les radiographies ce qui peut expliquer pourquoi il est difficile de distinguer les classes avec les méthodes de statistiques classiques. Un argument supplémentaire pour indiquer que les caractéristiques des radiographies sont variables est le faible nombre de variables qui sont identiques pour l’ensemble de l’échantillon (1321 soit 1.5%) après masquage alors qu’on aurait pu s’attendre à des zones masquées communes sur des régions non concernées par les poumons.

## Conclusion sur l’exploration des données

On ne constate donc pas de solution évidente à notre problème de classification avec une séparation dans un plan ce qui suggère que des modèles de classification classiques ne sont pas adaptés. Des solutions plus complexes seront donc à mettre en œuvre notamment avec l'emploi de modèle de Deep Learning.

</div>
"""
st.markdown(text_11, unsafe_allow_html=True)

