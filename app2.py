import os
import PyPDF2
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import base64

# Configuration de la page
st.set_page_config(
    page_title="Détection de Plagiat",
    page_icon="📚",
    layout="wide"
)
st.markdown("""
    <style>
       .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: white;
            background: linear-gradient(to right, #1f77b4, #4a90e2);
            padding: 1rem 0;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .stSidebar {
            background: linear-gradient(to bottom, #ffffff, #f0f2f6);
        
            }
        .stSidebar .css-1lcbmhc {
            color: black;  /* Couleur du texte */
            font-size: 1rem;
        }
               /* Fond de la sidebar */
        [data-testid="stSidebar"] {
            background: black;
        }
            
        
        /* Texte des options du menu */
        [data-testid="stSidebar"] .css-1v3fvcr {
            color: black;  /* Couleur du texte */
            font-size: 1rem;
        }
        
        /* Texte des options actives */
        [data-testid="stSidebar"] .css-qbe2hs {
            color: black;  /* Couleur des titres actifs */
            font-size: 1rem;
            font-weight: bold;
        }
        .stSidebar .css-17eq0hr {
            color: black;  /* Couleur des titres actifs */
            font-size: 1rem;
            font-weight: bold;
        }
        .dataframe tbody tr:hover {
            background-color: #f4f4f8;
        }
        .dataframe thead {
            background-color: #1f77b4;
            color: white;
        }
        .stProgress > div > div > div > div {
            background-color: #1f77b4;
        }
    </style>
""", unsafe_allow_html=True)


# 1. Fonction pour lire les documents (supporte txt, pdf, docx)
def preprocess_document(text):
    return ''.join(c.lower() for c in text if c.isalnum() or c.isspace())

def read_txt(file):
    return preprocess_document(file.read().decode("utf-8"))

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''.join(page.extract_text() for page in reader.pages)
    return preprocess_document(text)

def read_docx(file):
    doc = Document(file)
    text = ''.join(paragraph.text for paragraph in doc.paragraphs)
    return preprocess_document(text)

def read_uploaded_files(uploaded_files):
    documents = []
    filenames = []
    for file in uploaded_files:
        filenames.append(file.name)
        if file.name.endswith('.txt'):
            documents.append(read_txt(file))
        elif file.name.endswith('.pdf'):
            documents.append(read_pdf(file))
        elif file.name.endswith('.docx'):
            documents.append(read_docx(file))
    return documents, filenames

def analyze_mode(documents, filenames, mode):
    similarity_matrix = compute_similarity(documents)
    if mode == "Comparer 2 documents":
        st.subheader("Résultats : Comparaison entre deux documents")
        score = similarity_matrix[0, 1] * 100
        st.write(f"**Similarité Cosinus :** {score:.2f}%")
        if score > 75:
            st.success("Les documents semblent très similaires.")
        elif score > 50:
            st.warning("Les documents présentent des similarités modérées.")
        else:
            st.info("Les documents sont peu similaires.")
    elif mode == "Grouper n documents par similarité":
        st.subheader("Résultats : Grouper les documents par similarité")
        st.dataframe(pd.DataFrame(similarity_matrix, index=filenames, columns=filenames))
        st.write("### Heatmap des similarités")
        plot_similarity_heatmap(similarity_matrix, filenames)
        st.write("### Groupes de documents similaires(70%)")
        clustering_results = group_documents_by_similarity(similarity_matrix, filenames)
        st.write(clustering_results)
    return similarity_matrix


# 2. Calcul des similarités
def compute_similarity(documents):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    return cosine_similarity(tfidf_matrix)


# 3. Visualisation (Heatmap)
def plot_similarity_heatmap(similarity_matrix, filenames):
    plt.figure(figsize=(6, 4))
    sns.heatmap(similarity_matrix,
                 annot=True,
                   xticklabels=filenames,
                     yticklabels=filenames,
                       cmap='Blues',
                         fmt=".2f",
                         cbar_kws={'shrink': 0.75})
    plt.xticks(rotation=45, ha='right', fontsize=8)  # Réduire la taille des textes
    plt.yticks(fontsize=8)  # Réduire la taille des textes
    plt.tight_layout()  # Assurez-vous que tout est bien ajusté
    st.pyplot(plt)

def group_documents_by_similarity(similarity_matrix, filenames, threshold=0.7):
    groups = []
    used = set()
    for i, filename in enumerate(filenames):
        if i not in used:
            group = [filename]
            for j in range(len(similarity_matrix)):
                if i != j and similarity_matrix[i, j] >= threshold:
                    group.append(filenames[j])
                    used.add(j)
            groups.append(group)
    result = {f"Groupe {i + 1}": group for i, group in enumerate(groups)}
    return pd.DataFrame(dict([(k, pd.Series(v)) for k, v in result.items()]))

# 4. Section de téléchargement des fichiers
def file_upload_section(mode):
    st.header("Téléchargement des fichiers")
    uploaded_files = st.file_uploader(
        "📂 Téléchargez vos fichiers (formats pris en charge : .txt, .pdf, .docx)",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx']
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} fichiers téléchargés avec succès !")
        if st.button("⚙️ Analyser"):
            documents, filenames = read_uploaded_files(uploaded_files)
            if len(documents) < 2:
                st.error("Veuillez télécharger au moins 2 fichiers pour comparer.")
                return None
            
            # Calcul des similarités
            similarity_matrix = analyze_mode(documents,filenames,mode)
            
            # Sauvegarder les résultats dans `st.session_state`
            st.session_state['results'] = similarity_matrix
            st.session_state['filenames'] = filenames
            st.success("Analyse terminée !")
            return True
    return False

def create_navigation():
    """Crée la barre de navigation"""
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown('<div class="main-header">Détection de Plagiat</div>', unsafe_allow_html=True)
    
    menu = st.sidebar.radio(
        "Navigation",
        ["Accueil", "Charger des fichiers"],
        key="navigation_menu"
    )
    return menu

def create_footer():
    """Crée le pied de page"""
    st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background: linear-gradient(to right, #1f77b4, #4a90e2);
            color: white;
            padding: 1rem;
            text-align: center;
        }
        .footer a {
            color: white;
            font-weight: bold;
            text-decoration: none;
        }
    </style>
    <div class="footer">
        📚 Développé pour le projet INF4268 | Par BOKAGNE,ONDOA,PAMOM
        <a href="pamomyusuf@institutsaintjean.org">💬 Support technique</a> | 
    </div>
""", unsafe_allow_html=True)


# 5. Afficher les résultats
def display_results():
    st.header("Résultats de l'analyse")
    
    similarity_matrix = st.session_state['results']
    filenames = st.session_state['filenames']
    
    # Afficher la matrice sous forme de tableau
    st.subheader("Matrice de Similarité Cosinus")
    results_df = pd.DataFrame(similarity_matrix, index=filenames, columns=filenames)
    st.dataframe(results_df)
    
    # Heatmap
    st.subheader("Heatmap des Similarités")
    plot_similarity_heatmap(similarity_matrix, filenames)
    
    # Export des résultats
    st.subheader("Exporter les résultats")
    csv = results_df.to_csv()
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="resultats_similarite.csv">Télécharger les résultats en CSV</a>'
    st.markdown(href, unsafe_allow_html=True)


# 6. Navigation principale
def main():
    menu = create_navigation()
    
    if menu == "Accueil":
        st.write("Bienvenue dans l'application de détection de plagiat! Utilisez le menu pour commencer.")
    
    elif menu == "Charger des fichiers":
        st.subheader("Choisissez un mode d'analyse")
        mode = st.radio(
            "Mode d'analyse",
            ["Comparer 2 documents", "Grouper n documents par similarité"]
        )
        if file_upload_section(mode):
            st.experimental_set_query_params()
    
    elif menu == "Résultats":
        if 'results' in st.session_state:
            display_results()
        else:
            st.warning("Aucun résultat disponible. Veuillez analyser des fichiers d'abord.")
    
    create_footer()


if __name__ == "__main__":
    main()
