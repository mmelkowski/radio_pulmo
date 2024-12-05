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


header_text = """
<div style="text-align: justify;">

# Modélisation des données:
Un modèle de deep-learning a été dévéloppé afin de classifier les radiographies pulmonaires en quatre catégories.
    - COVID         : patient infecté par le COVID19
    - Normal        : radiographie d'un patient sans affection
    - Viral Pnumonia: patient souffrant de pneumonie virale
    - Lung Opacity  : patient présentant une opacité pulmonaire

 Le modèle avec Keras et TensorFlow à partir de l'architecture EfficientNetB4.
 Les données d'entrée sont les radiographies après masquage pour éviter tout biais lié à l'arrière plan.   
 La diversité du jeu de donnée a été amélioré via l'augmentation des données pour une meilleur généralisation.

 ## Classification des images
 Le modèle EfficientNetB4 a été choisi car il donnait les meilleurs résultats lors de nos tests préliminaires.
 Il s'agit d'un modèle réseau de neurones convolutifs introduit par Google en 2019.
 
 ### Le modèle EfficientNetB4 utilisé 
 Le schéma ci-dessous représente la classification qui a été réalisée à partir de notre jeu de données.

</div>
"""
st.markdown(header_text,unsafe_allow_html=True)
adaptedENB4_path = pathlib.Path('resources/modelisation/Adapted_ENB4.png')
st.image(str(adaptedENB4_path))

# Afficher le dataframe avec un bouton 
if st.button("Afficher le détail de l'architecture EfficientNetB4"):
    # Si le bouton est cliqué, afficher le jeu de données
    detailsENB4_path = pathlib.Path('resources/modelisation/Details_EfficientNetB4.png')
    st.image(str(detailsENB4_path))
    st.write("D'après Zhu et al., Frontiers in Medicine. 8. 10.3389/fmed.2021.626369.")


text1 = """
<div style="text-align: justify;">

 ### Performances du modèle
 Les graphiques ci-dessous représente l'évolution de la précision et de la fonction de perte au cours des epochs pour le jeu de données de validation et d'apprentissage.

 La matrice de confusion sur le jeu de données de validation est également présenté.
 La précision moyenne du modèle sur le jeu de données de validation est de 0.92 avec une précision de 0.93 et un recall de 0.90 sur la catégorie COVID qui nous intéresse particulièrement.
</div>
"""
st.markdown(text1,unsafe_allow_html=True)

# Afficher le dataframe avec un bouton 
if st.button("Afficher l'évolution de la fonction de perte pour la classification "):
    # Si le bouton est cliqué, afficher le jeu de données
    Loss_Accuracy_path = pathlib.Path('resources/modelisation/Loss_Accuracy.png')
    st.image(str(Loss_Accuracy_path))


# Afficher le dataframe avec un bouton 
if st.button("Afficher la matrice de confusion sur le jeu de données de test "):
    # Si le bouton est cliqué, afficher le jeu de données
    Confusion_path = pathlib.Path('resources/modelisation/Confusion_Matrix.png')
    st.image(str(Confusion_path))


text2 = """
<div style="text-align: justify;">

 ## Génération de masques 
 
 La classification des images ayant été réalisée sur données masquées, nous avons développé un modèle de segmentation permettant de générer un masque pour toute nouvelle radiographie pulmonaire.
 Le modèle a été entrainé sur les masques et les images du jeu de données.

 ### Modèle de segmentation U-NET
 Le modèle utilisé est le modèle de segmentation de type U-NET qui est le plus utilisé en imagerie médicale.
 Il est composé de 10 blocs et 27 couches pour un total de 7.8 millions de paramètres.

 Le schéma ci-dessous représente la segmenttion qui a été réalisée à partir de notre jeu de données.

 </div>
"""

st.markdown(text2,unsafe_allow_html=True)
adaptedUNET_path = pathlib.Path('resources/modelisation/Adapted_UNET.png')
st.image(str(adaptedUNET_path))

text3 = """
<div style="text-align: justify;">

 ### Performances du modèle
 Les graphiques ci-dessous représente l'évolution de la précision et de la fonction de perte au cours des epochs pour le jeu de données de validation et d'apprentissage.

 La précision moyenne sur le jeu de données de test est de 0.992 ce qui confirme que nous pouvons l'utiliser pour la génération de masques.
</div>
"""
st.markdown(text3,unsafe_allow_html=True)


# Afficher le dataframe avec un bouton 
if st.button("Afficher l'évolution de la fonction de perte pour la segmentation "):
    # Si le bouton est cliqué, afficher le jeu de données
    Loss_Accuracy_UNET_path = pathlib.Path('resources/modelisation/Loss_Accuracy_UNET.png')
    st.image(str(Loss_Accuracy_UNET_path))
