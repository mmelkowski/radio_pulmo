import streamlit as st
import pandas as pd
import pathlib

# import custom Navigation bar
from modules.nav import Navbar


#config:
st.set_page_config(page_title="Radio-Pulmo Modélisation", page_icon="resources/x-ray.ico")
path_to_resources = pathlib.Path("src/streamlit/resources")

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
st.markdown(header_text, unsafe_allow_html=True)
adaptedENB4_path = path_to_resources / "modelisation" / "Adapted_ENB4.png"
st.image(str(adaptedENB4_path))

# Afficher le dataframe avec un bouton
if st.button("Afficher le détail de l'architecture EfficientNetB4"):
    # Si le bouton est cliqué, afficher le jeu de données
    detailsENB4_path = path_to_resources / "modelisation" / "Details_EfficientNetB4.png"

    textEN = """
    Le modèle EfficientNetB4 se décompose en sept grand blocs, composés de sous-blocs eux-même composé de modules.
    Chaque module comprends une combinaison de multiplication, de convolution et d'élimination (drop-out).
    L'illustration ci-dessous est reprise d'une adaptation de Vardan Agarwal disponible [ici](https://towardsdatascience.com/complete-architectural-details-of-all-efficientnet-models-5fd5b736142).
    Le modèle est décrit dans l'[article](https://arxiv.org/abs/1905.11946) et implémenté dans Keras.
    """
    st.markdown(textEN, unsafe_allow_html=True)
    st.image(str(detailsENB4_path))

text1 = """
<div style="text-align: justify;">

 ### Performances du modèle
 Les graphiques ci-dessous représente l'évolution de la précision et de la fonction de perte au cours des epochs pour le jeu de données de validation et d'apprentissage.

 La matrice de confusion sur le jeu de données de validation est également présenté.
 La précision moyenne du modèle sur le jeu de données de validation est de 0.92 avec une précision de 0.93 et un recall de 0.90 sur la catégorie COVID qui nous intéresse particulièrement.
</div>
"""
st.markdown(text1, unsafe_allow_html=True)

# Afficher le graph d'évolution de la loss et de l'accuracy
Loss_Accuracy_path = path_to_resources / "modelisation" / "Loss_Accuracy.png"
st.image(str(Loss_Accuracy_path))


# Afficher la matrice de confusion
Confusion_path = path_to_resources / "modelisation" / "Confusion_Matrix.png"
st.image(str(Confusion_path))

# Afficher le rapport de classification

data_report = {
    "Class": [
        "COVID",
        "Lung_Opacity",
        "Normal",
        "Viral_Pneumonia",
        "accuracy",
        "macro avg",
        "weighted avg",
    ],
    "Precision": [0.93, 0.93, 0.90, 0.98, "", 0.94, 0.92],
    "Recall": [0.90, 0.84, 0.97, 0.94, 0.92, 0.91, 0.92],
    "F1-Score": [0.91, 0.88, 0.94, 0.96, "", 0.92, 0.92],
    "Support": [362, 602, 1019, 134, 2117, 2117, 2117],
}

# Conversion en DataFrame
df_report = pd.DataFrame(data_report).set_index("Class")

# Affichage dans Streamlit

clf_report = st.checkbox("Afficher le rapport de classification")
if clf_report:
    st.table(df_report)


text2 = """
<div style="text-align: justify;">

 ### Exemples de prédiction 
 
 La figure ci-dessous présente les résultats de classification ainsi que la visualisation des zones les plus impactantes pour la prédiction pour quatre images du jeu de données de test.

 </div>
"""

st.markdown(text2, unsafe_allow_html=True)
ex_pred_path = path_to_resources / "modelisation" / "exemple_prediction.png"
st.image(str(ex_pred_path))

text3 = """
<div style="text-align: justify;">

 ## Génération de masques 
 
 La classification des images ayant été réalisée sur données masquées, nous avons développé un modèle de segmentation permettant de générer un masque pour toute nouvelle radiographie pulmonaire.
 
 Le modèle a été entrainé sur 8690 (3000 max par catégorie) masques et images du jeu de données initial.

 ### Modèle U-NET
 Le modèle utilisé pour la segmentation est de type U-NET qui est actuellement le plus utilisé en imagerie médicale.
 Il est composé de 10 blocs et 27 couches pour un total de 7.8 millions de paramètres.

 Le schéma ci-dessous représente la modélisation réalisée sur notre jeu de données.

 </div>
"""

st.markdown(text3, unsafe_allow_html=True)
adaptedUNET_path = path_to_resources / "modelisation" / "Adapted_UNET.png"
st.image(str(adaptedUNET_path))

text3 = """
<div style="text-align: justify;">

 ### Performances du modèle
 Les graphiques ci-dessous représentent l'évolution de la précision et de la fonction de perte au cours des epochs pour le jeu de données d'apprentissage et de validation.

 La précision moyenne sur le jeu de données de test est de 0.992 ce qui confirme que nous pouvons l'utiliser pour la génération de masques à partir de nouvelles images.
</div>
"""
st.markdown(text3, unsafe_allow_html=True)

# Si le bouton est cliqué, afficher le jeu de données
Loss_Accuracy_UNET_path = path_to_resources / "modelisation" / "Loss_Accuracy_UNET.png"
st.image(str(Loss_Accuracy_UNET_path))


data_unet_report = {
    "Dataset": ["Training", "Validation", "Test"],
    "Sample Size": [7038, 783, 869],
    "Pixel-wise Accuracy": [0.9958, 0.9917, 0.9918],
    "Dice Coefficient": [0.9911, 0.9822, 0.9822],
    "Dice Loss": [0.0089, 0.0178, 0.0178],
}

# Conversion en DataFrame
dfunet_report = pd.DataFrame(data_unet_report).set_index("Dataset")

# Affichage dans Streamlit

unet_report = st.checkbox(
    "Afficher les métriques sur le jeu de données d'apprentissage, test et validation"
)
if unet_report:
    st.table(dfunet_report)


text3 = """
<div style="text-align: justify;">

 ## Conclusion
 Ces deux étapes de modélisation permettent de génerer des masques puis de diagnostiquer automatiquement des radiographies pulmonaires.
 Le modèle de segmentation a une précision supérieure à 0.99 ce qui est très satisfaisant.
 Le modèle de classification a quant à lui une précision de 0.92 ce qui est conforme à l'[état de l'art](https://arxiv.org/pdf/2003.09871).
 
 Il serait pertinent d'explorer l'efficacité des modèles de type Transformer afin d'améliorer encore cette précision. 
 Cependant, leur utilisation a révélé des coûts computationnels trop élevés au regard des ressources informatiques disponibles.
 
 Pour aller plus loin, il serait également intéressant de disposer d'annotations sur le degré de sévérité de l'atteinte pulmonaire, 
 ce qui permettrait d'enrichir notre application avec cette prédiction.

 """
st.markdown(text3, unsafe_allow_html=True)
