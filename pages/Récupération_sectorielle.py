# Initialisation

# Interface et graphiques
import streamlit as st
import plotly.express as px

# Data management
import pandas as pd

# NLP
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from mtranslate import translate
import spacy
import re
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Mathématiques
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from Levenshtein import distance



df_villes = pd.read_csv("Data/cities_V2.csv", encoding='latin1',sep=";")


pd.set_option('display.max_colwidth', None)

st.markdown("#  <center>  :pushpin: Récupération sectorielle  :pushpin:</center> ", unsafe_allow_html=True)  

st.markdown("L'objectif ici est de transformer un texte en information géographique. Les retours usagers n'étant pas obligatoirement classifiés selon la localité géographique, cela permet d'avoir une répartition spatiale automatisée.")


st.markdown("<h2> <b> Concept : </b> </h2>", unsafe_allow_html=True)  

# Charger le modèle spacy pour la langue française
nlp = spacy.load('fr_core_news_sm')


# Texte à analyser

st.markdown("Si un texte est déposé dans l'espace en dessous alors il peut se transformer en carte géographique.")

texte = st.text_input("Retour usager :", "Le train de Paris vers Amiens est toujours à l'heure. Je suis content, il passe par Longueau.")
temp=texte


# Appliquer le modèle pour extraire les entités nommées
doc = nlp(texte)
villes = []

for ent in doc.ents:
    if ent.label_ == "LOC":
        villes.append(ent.text)
        


# Afficher les noms de villes trouvés
#print("Noms de villes françaises trouvés : ", villes)
st.markdown(villes, unsafe_allow_html=True)




# Enlever les lignes avec des valeurs manquantes dans les colonnes "latitude" et "longitude"
df_villes.dropna(subset=["latitude", "longitude"], inplace=True)

# Construire le dataframe "resultat" qui contient toutes les villes similaires et leurs coordonnées
villes_similaires = []
for ville_input in villes:
    # Calculer la distance de Levenshtein entre la ville d'entrée et chaque ville dans la base de données
    df_villes["distance"] = df_villes["label"].apply(lambda x: distance(x.lower(), ville_input.lower()))

    # Normaliser les distances pour obtenir une similarité entre 0 et 1
    max_distance = df_villes["distance"].max()
    df_villes["similarite"] = 1 - (df_villes["distance"] / max_distance)

    # Trier les villes par ordre de similarité décroissante
    df_villes = df_villes.sort_values(by="similarite", ascending=False)

    # Ajouter toutes les villes similaires à la liste "villes_similaires"
    top_villes = df_villes.head(1).copy()
    top_villes.loc[:, "Villes"] = ville_input
    villes_similaires.append(top_villes[["label", "latitude", "longitude", "Villes"]])


# Concaténer tous les dataframes dans un seul dataframe "resultat"
resultat = pd.concat(villes_similaires, ignore_index=True)

# Afficher les villes similaires pour chaque ville trouvée dans le texte
#for ville_input in villes:
    #st.markdown(f"Ville similaire à {ville_input} : {resultat[resultat['Villes'] == ville_input]['label'].tolist()}")


# Afficher toutes les villes similaires sur une carte avec Plotly
fig = px.scatter_mapbox(
    resultat,
    lat="latitude",
    lon="longitude",
    hover_name="label",
    zoom=6,
    mapbox_style="carto-positron",
    color="Villes"
)

st.plotly_chart(fig)

# Afficher les villes similaires pour chaque ville trouvée dans le texte
for ville_input in villes:
    ville_similaire = resultat[resultat["Villes"] == ville_input]
    ville_label = ville_similaire["label"].tolist()[0]
    ville_distance = df_villes[df_villes["label"] == ville_label]["distance"].tolist()[0]
    ville_similarite = df_villes[df_villes["label"] == ville_label]["similarite"].tolist()[0]
    st.markdown(' '.join([
        f"Ville similaire à {ville_input} : {ville_label}",
        f"Score de distance : {ville_distance}",
        f"Score de similarité : {ville_similarite}"
    ]))

    
st.markdown("<h2> <b> Application : </b> </h2>", unsafe_allow_html=True)  

def Extraction_ville(texte):
  """Extrait les villes d'un texte"""
  doc = nlp(texte)
  #print(doc)
  villes = []
  for ent in doc.ents:
    if ent.label_ == "LOC" and "hdf" not in ent.text.lower():
        cleaned_text = re.sub(r'\W', '', ent.text)
        villes.append(cleaned_text )
  return(villes)
    
def Coordonnees_geographiques(villes,df_villes) :
  """Prends une liste et retourne des coordonnées géographiques associées à chaques villes"""
  villes_similaires = []

  for ville_input in villes:
    # Calculer la distance de Levenshtein entre la ville d'entrée et chaque ville dans la base de données
    df_villes["distance"] = df_villes["label"].apply(lambda x: distance(x.lower(), ville_input.lower()))

    # Normaliser les distances pour obtenir une similarité entre 0 et 1
    max_distance = df_villes["distance"].max()
    df_villes["similarite"] = 1 - (df_villes["distance"] / max_distance)

    # Trier les villes par ordre de similarité décroissante
    df_villes = df_villes.sort_values(by="similarite", ascending=False)

    # Ajouter toutes les villes similaires à la liste "villes_similaires"
    top_villes = df_villes.head(1).copy()
    top_villes.loc[:, "Villes"] = ville_input
    villes_similaires.append(top_villes[["label", "latitude", "longitude"]]) #["label", "latitude", "longitude", "Villes"]

  return(villes_similaires)

df_texte = pd.read_csv("Data/Retours_usagers.csv", sep=';')
#liste_df = list(df.Contenu_retour)
#df_texte = liste_df
df_texte=df_texte[["Contenu_retour"]]
df_texte['Villes']=df_texte["Contenu_retour"].apply(lambda x: Extraction_ville(x))
st.dataframe(df_texte)
df_texte['Coordonnees']=df_texte["Villes"].apply(lambda x: Coordonnees_geographiques(x,df_villes))
#df_texte

liste_ville=[]
liste_latitude=[]
liste_longitude=[]


for i in df_texte["Coordonnees"]:
  #print(i)
  
  for j in i :
    #print(j["latitude"])
    liste_ville.append(j["label"].values[0])
    liste_latitude.append(j["latitude"].values[0])
    liste_longitude.append(j["longitude"].values[0])

df_carte=pd.DataFrame({'Nom_ville': liste_ville,'latitude': liste_latitude,'longitude': liste_longitude})
df_carte['size'] = df_carte.groupby('Nom_ville')['Nom_ville'].transform('size')
df_carte = df_carte.drop_duplicates(subset='Nom_ville')
df_carte=df_carte[df_carte['size']>1]

fig = px.scatter_mapbox(
    df_carte,
    lat="latitude",
    lon="longitude",
    hover_name="Nom_ville",
    zoom=5,
    mapbox_style="carto-positron",
    size="size"
)
st.plotly_chart(fig)

