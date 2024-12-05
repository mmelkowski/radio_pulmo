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

# Modélisation des données
Un modèle de deep-learning a été dévéloppé à partir de notre jeu de données afin de classifier les radiographies pulmonaires en quatre catégories:
- COVID             : patient infecté par le COVID19
- Normal            : radiographie d'un patient sans affection
- Viral Pneumonia   : patient souffrant de pneumonie virale
- Lung Opacity      : patient présentant une opacité pulmonaire

 Le modèle a été développé avec Keras et TensorFlow à partir de l'architecture EfficientNetB4.

 Les données d'entrée sont les radiographies après masquage pour éviter tout biais lié à l'arrière plan.
 
 Leur diversité ont été améliorée via l'augmentation de données pour une meilleure généralisation.

 ## Classification des images
 Le modèle EfficientNetB4 a été choisi car il donnait les meilleurs résultats lors de nos tests préliminaires.
 Il s'agit d'un modèle réseau de neurones convolutifs d'environ 300 couches pour 19 millions de paramètres introduit par Google en 2019.
 
 ### Modèle EfficientNetB4 
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

# Afficher le graph d'évolution de la loss et de l'accuracy
Loss_Accuracy_path = pathlib.Path('resources/modelisation/Loss_Accuracy.png')
st.image(str(Loss_Accuracy_path))


# Afficher la matrice de confusion
Confusion_path = pathlib.Path('resources/modelisation/Confusion_Matrix.png')
st.image(str(Confusion_path))


text2 = """
<div style="text-align: justify;">

 ## Génération de masques 
 
 La classification des images ayant été réalisée sur données masquées, nous avons développé un modèle de segmentation permettant de générer un masque pour toute nouvelle radiographie pulmonaire.
 
 Le modèle a été entrainé sur les masques et les images du jeu de données initial.

 ### Modèle U-NET
 Le modèle utilisé pour la segmentation est de type U-NET qui est actuellement le plus utilisé en imagerie médicale.
 Il est composé de 10 blocs et 27 couches pour un total de 7.8 millions de paramètres.

 Le schéma ci-dessous représente la modélisation réalisée sur notre jeu de données.

 </div>
"""

st.markdown(text2,unsafe_allow_html=True)
adaptedUNET_path = pathlib.Path('resources/modelisation/Adapted_UNET.png')
st.image(str(adaptedUNET_path))

text3 = """
<div style="text-align: justify;">

 ### Performances du modèle
 Les graphiques ci-dessous représentent l'évolution de la précision et de la fonction de perte au cours des epochs pour le jeu de données d'apprentissage et de validation.

 La précision moyenne sur le jeu de données de test est de 0.992 ce qui confirme que nous pouvons l'utiliser pour la génération de masques à partir de nouvelles images.
</div>
"""
st.markdown(text3,unsafe_allow_html=True)

# Si le bouton est cliqué, afficher le jeu de données
Loss_Accuracy_UNET_path = pathlib.Path('resources/modelisation/Loss_Accuracy_UNET.png')
st.image(str(Loss_Accuracy_UNET_path))


text3 = """
<div style="text-align: justify;">

 ## Conclusion
 Ces deux étapes de modélisation permettent de génerer des masques puis de diagnostiquer automatiquement des radiographies pulmonaires.
 Le modèle de segmentation a une précision supérieure à 0.99 ce qui est très satisfaisant.
 Le modèle de classification a quant à lui une précision de 0.92 ce qui est conforme à l'état de l'art.

 Il pourrait être intéressant de tester l'efficacité de modèle de type transformers pour améliorer encore cette précision mais ils 
 se sont avérés trop coûteux en ressources informatiques pour les équipements à notre disposition. 
</div>
"""
st.markdown(text3,unsafe_allow_html=True)