import streamlit as st


def Navbar():
    """Creates a navigation bar for a Streamlit multipage application.

    This function creates a sidebar containing the application title, logo,
    and navigation links to other pages defined in the Streamlit app.

    Returns:
        None
    """
    with st.sidebar:
        st.sidebar.title("Radiographie Pulmonaire")
        st.sidebar.image("src/streamlit/resources/x-ray.png", use_container_width=True)
        st.page_link("app.py", label="Application", icon="🚀")  # 🚀🔥
        st.page_link("pages/context.py", label="Contexte", icon="🧩")  # 📚🖼️
        st.page_link(
            "pages/data_discovery.py", label="Découverte des données", icon="🔍"
        )
        st.page_link("pages/acp.py", label="ACP sur les données", icon="🎯")  # 🧮🔵
        st.page_link("pages/model.py", label="Modélisation", icon="📊")  # 🤖🛠️

        # Crédit
        bottom_text = """
        <div style="font-size: 14px; color: gray; font-style: italic; text-align: center; margin-top: 20px;">
        Cette application a été développée par 
            <br>
            <a href="https://www.linkedin.com/in/chris-hozé-007901a5" target="_blank" style="color: #0073e6;">Chris Hozé</a> 
            et 
            <a href="https://www.linkedin.com/in/mickael-melkowski/" target="_blank" style="color: #0073e6;">Mickaël Melkowski</a>
            <br> dans le cadre de notre formation en DataScience réalisée avec DataScientest.
            <br>
            <br>L'ensemble des scripts et modèles sont disponibles sur le dépot 
            <a href="https://github.com/mmelkowski/radio_pulmo/" target="_blank" style="color: #0073e6;">github</a> du projet.
        </div>
        """
        # Affichage du texte dans la sidebar
        st.sidebar.markdown(bottom_text, unsafe_allow_html=True)
