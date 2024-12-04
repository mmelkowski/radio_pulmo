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
# Analyse en composante principale:

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
    # Vérifier que l'indice i est dans les limites de explained_variance
        
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
    value=10,  # Valeur par défaut
    step=2
)

fig = plot_var(pca_var, num_components)

# Afficher le countplot dans Streamlit
st.plotly_chart(fig)


text_10 = """
<div style="text-align: justify;">

### Visualisation des images sur les composantes de l’ACP
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

#nb_composante=[f'PC{i+1}' for i in range(pca_df.shape[1]-3)]

# Choisir les 2 ou 3 composantes à afficher
st.sidebar.title("Choix des composantes")
PCA_choice = st.multiselect(
    "Choisissez deux ou trois composantes principales à visualiser",
    options=range(pca_df.shape[1]-3), 
    default=[0, 1, 2], 
    max_selections=3
)

# # # Vérifier si l'utilisateur a sélectionné des composantes
# if PCA_choice:
#     selected_comps = [f"PC{i+1}" for i in PCA_choice]
#     st.write("Composantes choisies :", selected_comps)
# else:  
#     st.write("Aucune composante sélectionnée. Veuillez en choisir au moins une.")

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




# PCA_choice = st.radio(
#     "**Choix des composantes de l'ACP:**",
#     ["PCA1 - PCA2", "PCA3 - PCA4"]
# )

# if PCA_choice == "PCA1 - PCA2":
#     st.markdown("""<div style="text-align: justify;">On visualise les coordonnées des échantillons sur les plans 1 et 2 pour voir s'il est possible de séparer les catégories selon un plan. On étudie également la répartition par source de données pour identifier une éventuelle source de biais. :</div>""", unsafe_allow_html=True)
#     st.image("resources/decouverte_donnees/ACP/ACP_1.png")
#     st.image("resources/decouverte_donnees/ACP/ACP_2.png")
# else:
#     st.markdown("""<div style="text-align: justify;">On visualise les coordonnées des échantillons sur les plans 3 et 4 pour voir s'il est possible de séparer les catégories selon un plan. On étudie également la répartition par source de données pour identifier une éventuelle source de biais. :</div>""", unsafe_allow_html=True)
#     st.image("resources/decouverte_donnees/ACP/ACP_3.png")
#     st.image("resources/decouverte_donnees/ACP/ACP_4.png")

text_11 = """
<div style="text-align: justify;">


### Interprétation

Les axes ne permettent pas de séparer les échantillons par groupe, et on identifie pas de sous-population en fonction de la source de données. 
L’orientation, le niveau de zoom et la taille des poumons diffèrent selon les patients et les radiographies ce qui peut expliquer pourquoi 
il est difficile de distinguer les classes avec l'analyse en composantes principales. 

On ne constate donc pas de solution évidente à notre problème de classification avec une séparation dans un plan ce qui suggère que des modèles de classification classiques ne sont pas adaptés. 
Des solutions plus complexes pouvant prendre en compte un grand nombre de features comme le deep-learning seront donc à mettre en œuvre.

</div>
"""
st.markdown(text_11, unsafe_allow_html=True)

