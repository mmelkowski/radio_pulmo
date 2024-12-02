import streamlit as st

# import custom Navigation bar
from modules.nav import Navbar
Navbar()


intro_text = """
# Prise en main et découverte des données:
## Visualisation de quelques images du dataset:
"""
st.markdown(intro_text)
st.image("resources/decouverte_donnees/showcase.png")

text_2 = """
<div style="text-align: justify;">

Les images sont des radiographies pulmonaires provenant de différentes sources, on remarque rapidement que seule une partie de l'image nous intéressera puisqu'on s'intéressera aux poumons alors que l'image laisse notamment la colonne vertébrale et le cœur. Il faudra donc appliquer les masques qui ont été précalculés.

Les données sont en échelle de gris, une importation en couleur donne trois valeurs identiques pour les différents canaux. Chaque image correspond donc à une matrice de 299*299. Ci-dessous, un exemple d’image sur les trois canaux RGB.
</div>"""
st.markdown(text_2,unsafe_allow_html=True)
st.image("resources/decouverte_donnees/x-ray_rgb.png")

text_3 = """
<div style="text-align: justify;">

Les seules métadonnées disponibles concernent l’origine des radiographies. Aucune information n’est disponible sur les patients malgré leur intérêt évident pour ce type d’ étude.

## Pre-processing des données

L'ensemble des images ainsi que les métadonnées ont été importées.

Dans un premier temps, on applique un preprocessing simple sur les données : 

- Import
- Redimensionnement du masque à la taille de l'image de 256x256 à 299x299
- Application du masque

Un jeu de données tabulaire a été créé pour les analyses exploratoires, il contient 21165 lignes et 89406 colonnes correspondant aux métadonnées et les 89400 (299x299) pixels. Ce jeu de données sera utilisé pour l'exploration.
</div>
"""
st.markdown(text_3,unsafe_allow_html=True)
st.image("resources/decouverte_donnees/table.png")

text_4 = """
<div style="text-align: justify;">

## Statistiques descriptives sur les données
### Répartition des données par catégorie

La première analyse réalisée est une visualisation du nombre d'images par catégorie. 
</div>
"""
st.markdown(text_4, unsafe_allow_html=True)
st.image("resources/decouverte_donnees/barplot_disease.png")

text_5 = """
<div style="text-align: justify;">

On observe que nous n’avons majoritairement des données de radio pulmonaires pour les patients ne présentant pas d’affection. Les données de pneumonie virales sont quant à elles minoritaires.

### Répartition des données par source et par catégorie.

Les images proviennent de plusieurs sources différentes. On cherche à visualiser la répartition des catégories par groupe afin d’identifier un éventuel biais.

</div>
"""
st.markdown(text_5, unsafe_allow_html=True)
st.image("resources/decouverte_donnees/data_origin.png")

text_6 = """
<div style="text-align: justify;">

On remarque que les données pour la catégorie COVID proviennent de multiples sources. Les données “Lung Opacity” proviennent d’une source unique, idem pour les données “Viral Pneumonia”. Pour chacune de ses sources des données de catégorie “Normal” ont également été fournies ce qui devrait limiter la source de biais.

### Visualisation des différences entre catégories.

Dans une première approche, on crée une nouvelle variable contenant les valeurs moyennes pour l’ensemble des pixels pour chaque catégorie après masquage.
Le tableau suivant résume les différences observées sur les valeurs des pixels, en fonction des jeux de données.

| Catégorie  | COVID | Lung Opacity | Normal | Viral Pneumonia |
| ---------- | ----- | ------------ | ------ | --------------- |
| Moyenne    | 32,86 | 25,16        | 24,34  | 27,58           |
| Ecart Type | 32,94 | 29,31        | 32,22  | 37,52           |
| Minimum    | 0,00  | 0,00         | 0,00   | 0,00            |
| Médiane    | 19,41 | 10,64        | 5,18   | 5,54            |
| Maximum    | 99,88 | 100,69       | 106,73 | 120,63          |

On peut également les observer plus visuellement à l’aide de la boîte de dispersion ci-dessous:

</div>
"""
st.markdown(text_6, unsafe_allow_html=True)
st.image("resources/decouverte_donnees/boxplot.png")

text_7 = """
<div style="text-align: justify;">

Il semble que la moyenne des données COVID soient plus élevée, ce qui suggère des images plus blanches et donc une cohérence avec la plus grande opacité attendue.

Les données semblent plus extrêmes dans la catégorie Viral Pneumonia.

Une autre possibilité de visualisation est de tracer le nuage de point de la valeur moyenne du pixel en ordonnées et les pixels en abscisse et la moyenne en fonction de l’écart-type.

</div>
"""
st.markdown(text_7, unsafe_allow_html=True)
st.image("resources/decouverte_donnees/relplot.png")

st.markdown("""<div style="text-align: justify;">On visualise également l’écart-type en fonction de la moyenne pour chacune des catégories.</div>""", unsafe_allow_html=True)
st.image("resources/decouverte_donnees/dispersion.png")

text_8 = """
<div style="text-align: justify;">

Dans les deux représentations, on remarque que le comportement diffère en fonction des catégories. On distingue des formes plus sombres dans le cas des catégories COVID et Lung Opacity. Les données semblent également plus dispersées pour ces catégories.
Pour mieux visualiser ces différences, on trace l’image moyenne par groupe.

## Images moyennes par type et par source

Un autre axe d’exploration est de procéder au calcul de l’image moyenne par type et par sources pour tenter d’y déceler un biais.

### Visualisation

</div>
"""
st.markdown(text_8, unsafe_allow_html=True)
st.image("resources/decouverte_donnees/mean.png")

text_9 = """
<div style="text-align: justify;">

### Interprétation

Il semble y avoir un biais sur les analyses radio pulmonaires liées à la pneumonie virale : le cœur est proportionnellement plus important et l’application du masque efface la partie inférieure du poumon gauche. Ce biais est lié à l’origine des données issus du dataset “pneumonia-chestxray”. 

Sans certitude, on émet l’hypothèse à la vue des images que ces données correspondent à des radios pulmonaires d’enfants.

Une partie des données issues de ce jeu de données étant également de catégorie “Normal” on retrouve le même phénomène de façon moins marquée sur la catégorie normale.

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

