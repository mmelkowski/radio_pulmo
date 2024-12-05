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
# Analyse en composantes principales

Une ACP a été réalisé afin de poursuivre l'exploration des données pour identifier un éventuel biais, mais également réaliser un premier essai de classification pour déterminer le type de modèle à utiliser.

"""
st.markdown(titre_text)

text_1 = """

### Part de variance expliquée par les composantes de l’ACP

Il faut 874 composantes pour extraire 95 pour cent de la variance initiale.
Avec 20 composantes, on atteint 50.2 pour cent de variance expliquée. 
On observe donc une forte hétérogénéité.

La figure ci-dessous donne la variance cumulée en fonction du nombre de composantes. 

</div>
"""
st.markdown(text_1, unsafe_allow_html=True)

# Charger le dataset contenant uniquement label, source, moyenne et std par image
@st.cache_data
def load_data():
    var_pca_path = pathlib.Path('resources/pca_variance.pkl')
    df = pd.read_pickle(var_pca_path)
    return df

# Charger les données
pca_var = load_data()

def plot_var(pca_var, i):
        
    pca_var = pca_var.head(i)

    # Créer un DataFrame pour utiliser avec Plotly
    x_values = pca_var["Component"]
    y_values = pca_var["Explained Variance"]
    cumulative_values = y_values.cumsum()
    
    # Créer le graphique avec Plotly Express
    fig = px.bar(
        x=x_values,
        y=cumulative_values,
        title="Évolution de la variance expliquée par les composantes principales",
        labels={'x': 'Numéro de la composante principale', 'y': 'Variance cumulée'},
        text=y_values.round(3)
    )

    fig.add_annotation(
        x=0.5, y=1.1, text=f"Variance cumulée: {cumulative_values.iloc[-1]:.3f}", showarrow=False,
        font=dict(size=18, color="black"), align="left", xref="paper", yref="paper"
    )

    fig.update_traces(texttemplate='%{text:.3f}') 
    # Afficher le graphique
    return fig


num_components = st.slider(
    "Choisissez le nombre de composantes principales à afficher",
    min_value=1,
    max_value=pca_var.shape[0],  # Nombre maximum de composantes dans pca_var
    value=int(pca_var.shape[0]/2),  # Valeur par défaut
    step=1
)

fig = plot_var(pca_var, num_components)

# Afficher le countplot dans Streamlit
st.plotly_chart(fig)


text_10 = """
<div style="text-align: justify;">

### Répartition des images sur les composantes de l’ACP
</div>
"""


st.markdown(text_10, unsafe_allow_html=True)

# Charger le dataset contenant uniquement label, source, moyenne et std par image
@st.cache_data
def load_data():
    df_pca_path = pathlib.Path('resources/pca_df.pkl')
    df = pd.read_pickle(df_pca_path)
    return df

# Charger les données
pca_df = load_data()


# Choisir les 2 ou 3 composantes à afficher
st.sidebar.title("Choix des composantes")
PCA_choice = st.multiselect(
    "Choisissez deux ou trois composantes principales à visualiser",
    options=range(pca_df.shape[1]-3), 
    default=[0, 1, 2], 
    max_selections=3
)

# Demander à l'utilisateur sur quoi il veut se baser pour faire le countplot
options = ['label', 'source']
supp = st.selectbox("Choisir l'information à utiliser pour les boites de dispersion ", options)


def plot_pca(num_components, supp = "label"):
    #total_var = pca.explained_variance_ratio_[num_components].sum() * 100
    supp = pca_df[supp]
    cats = supp.unique()

    if len(num_components) == 3:
        
        fig = px.scatter_3d(

            pca_df, x = num_components[0], y =  num_components[1], z =  num_components[2], color = supp,
            #title=f'Total Explained Variance: {total_var:.2f}%',
            
        )
        fig.update_layout(scene=dict(xaxis=dict(backgroundcolor='darkgray', gridcolor='gray', tickcolor='white', title = "PC_" +  str(num_components[0])),
                yaxis=dict(backgroundcolor='darkgray', gridcolor='gray', tickcolor='white',  title = "PC_" +  str(num_components[1])),
                zaxis=dict(backgroundcolor='darkgray', gridcolor='gray', tickcolor='white', title = "PC_" +  str(num_components[2] ))))
        fig.update_traces(marker=dict(size=2))

    if len(num_components) == 2:
        fig = px.scatter(

            pca_df, x = num_components[0], y =  num_components[1], color = supp,
            #title=f'Total Explained Variance: {total_var:.2f}%',
           )
        fig.update_layout(scene=dict(xaxis=dict(backgroundcolor='slategray', gridcolor='lightgray', tickcolor='white', title = "PC_" +  str(num_components[0])),
                yaxis=dict(backgroundcolor='slategray', gridcolor='lightgray', tickcolor='white',  title = "PC_" +  str(num_components[1])))) 
        
        fig.update_traces(marker=dict(size=2))

    fig.update_layout(width=600, height=600)
    return fig


fig = plot_pca(PCA_choice, supp)

# Afficher le countplot dans Streamlit
st.plotly_chart(fig)


# Afficher le dataframe avec un bouton 
if st.button("Afficher les images de composantes de l'ACP"):
    # Ouvrir le dataframe
    pca_mean_img_path = pathlib.Path('resources/pca_mean_image.pkl')
    with open(pca_mean_img_path, 'rb') as g:
        pca_avg_img = pickle.load(g)

    #st.image(Image.fromarray(np.clip(pca_avg_img.iloc[0].values.reshape(256, 256) * 255, 0, 255).astype(np.uint8)), caption="Image Exemple")
    cols3 = st.columns(5)

        # Répartir les images sur 5 colonnes
    for i, (index, row) in enumerate(pca_avg_img.iterrows()):
        col_index = i % 5 

        img_array = row.values[:]
        img_array = img_array / np.mean(img_array) * 127

        img = img_array.reshape(256, 256)
        img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(np.uint8))
        img = img.convert('L')
        # enhancer = ImageEnhance.Contrast(img)
        # img = enhancer.enhance(1.5) 
        # # Augmenter la saturation
        # enhancer = ImageEnhance.Color(img)
        # img = enhancer.enhance(1.2)
        # Pour résoudre le problème d'alignement on diminue la longueur
        short_caption = "PCA" + str(i)
        cols3[col_index].image(img, caption=f'{short_caption}')
        #cols3[col_index].image(img, caption=f'{[row["label_source"]]}')



text_11 = """
<div style="text-align: justify;">


### Interprétation

Les composantes de l'ACP ne permettent pas de séparer les échantillons par groupe, et on identifie pas de sous-population en fonction de la source de données. 
Cette difficulté peut s'expliquer car l'orientation, le niveau de zoom et la taille des poumons diffèrent selon les patients et les radiographies.

On ne constate donc pas de solution évidente à notre problème de classification avec une séparation dans un plan. Ceci suggère que des modèles de classification classiques ne seront pas adaptés. 
Des solutions plus complexes pouvant prendre en compte un grand nombre de features comme le deep-learning seront donc à mettre en œuvre.

</div>
"""
st.markdown(text_11, unsafe_allow_html=True)

