import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import nltk
from nltk.corpus import stopwords
import spacy
nltk.download('stopwords')

pd.set_option('display.max_colwidth', None)

nltk.download('vader_lexicon')

st.markdown("#  <center> :scissors: Segmentation des retours :scissors:</center> ", unsafe_allow_html=True)  

st.markdown("L'idée ici est de proposer un outil qui permet d'anticiper les thèmes relatifs aux besoins usagers.")

st.markdown("<h2> <b> Concept : </b> </h2>", unsafe_allow_html=True) 

st.markdown("Les retours usagers seront regroupé en différentes catégories selon leur contenu  par l' identifiant de mots clés et de thèmes dominants dans chaque groupe.")

st.markdown(" Cette classification aide à synthétiser les données, permettant ainsi d'orienter les actions d'amélioration de l'expérience utilisateur.")


st.markdown("<h2> <b> Application : </b> </h2>", unsafe_allow_html=True)  

st.markdown("Essayons le concept sur une base test qui contient 25 tweets sur le hashtag #TERHdF.", unsafe_allow_html=True) 

df = pd.read_csv("Data/Retours_usagers.csv", sep=';')
liste_df = list(df.Contenu_retour)
texts = liste_df
texts=df.Contenu_retour

st.dataframe(df[["Contenu_retour"]])

       

num_clusters = 3
max_features = 10000
max_df = 0.8
min_df = 2

french_stop_words = list(set(stopwords.words('french')+["sncf","ter"]))
vectorizer = TfidfVectorizer(max_features=max_features, max_df=max_df, min_df=min_df, stop_words=french_stop_words)
text_vect = vectorizer.fit_transform(texts)

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(text_vect)

st.markdown("<b> Typologie cluster : </b> ", unsafe_allow_html=True)  

for i, cluster in enumerate(kmeans.cluster_centers_):
    st.markdown(f"Cluster {i+1} Mots majoritaires :")
    sorted_terms = [vectorizer.get_feature_names_out()[term_index] for term_index in cluster.argsort()[:-10 - 1:-1]]
    st.markdown(', '.join(sorted_terms))

    labels = kmeans.labels_
    unique, counts = np.unique(labels, return_counts=True)
    
st.markdown("<b> Effectifs : </b> ", unsafe_allow_html=True)  

st.markdown(dict(zip(unique, counts)))
# Calculer les pourcentages
percentages = (counts / sum(counts)) * 100

# Créer un dictionnaire avec les pourcentages
percentage_dict = dict(zip(unique, percentages))

# Afficher les pourcentages dans Streamlit
for theme, percentage in percentage_dict.items():
    st.markdown(f"Thème {theme}: {percentage:.2f}%")

st.markdown("<b> Interprétation : </b> ", unsafe_allow_html=True)  

st.markdown('Cluster 1 : Ce groupe semble se concentrer sur les problèmes liés aux retards et à la régularité. ')
st.markdown("Cluster 2 : Ce groupe refléte des discussions sur les trajets et les gares.")
st.markdown("Cluster 2 : Ce groupe évoque les outils numériques et la communication liée à l'offre TER en HdF.")


