# Environnement

# Interface et graphiques
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Data management
import pandas as pd
import datetime

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


st.set_page_config(initial_sidebar_state="collapsed")

df_villes = pd.read_csv("Data/cities_V2.csv",sep=";", encoding='latin1')
#pd.set_option('display.max_colwidth', None)
nlp = spacy.load('fr_core_news_sm')


def Extraction_ville(texte):
  """Extrait les villes d'un texte"""
  doc = nlp(texte)
  #print(doc)
  villes = []
  for ent in doc.ents:
    if ent.label_ == "LOC" and "hdf" not in ent.text.lower():
        cleaned_text = re.sub(r'\W', '', ent.text)
        if not any(char.isdigit() for char in cleaned_text)  and len(cleaned_text) > 3 and cleaned_text!="Bonjour" and cleaned_text!="bonjour" and cleaned_text!="JPFarandou": # skip if any digit is present in cleaned_text
          villes.append(cleaned_text)
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

# Clean the text data
def clean_text(text):
    text = re.sub(r'[^\w\s]','',text)
    text = text.replace('\x92', "'").replace('\x96', '-')
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.strip()
    return text

def formater_date(date):
    jours = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    mois = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]
    return "{} {} {} {}".format(jours[date.weekday()], date.day, mois[date.month - 1], date.year)

def translate_french_to_english(text):
    return translate(text, 'en', 'fr')

def translate_french_to_english_reverse(text):
    return translate(text, 'fr', 'en')

def translate_and_analyze_sentiment(text, source_lang='fr', target_lang='en'):
    sia = SentimentIntensityAnalyzer()
    # Traduction du texte
    translated_text = translate_french_to_english(text)
    # Calcul du score de sentiment
    sentiment_score = sia.polarity_scores(translated_text)
    return sentiment_score

# Fonction pour catégoriser les scores
def categorize_sentiment(score):
  if score['compound'] >= 0.6:
      return "très positif"
  elif score['compound'] > 0.25 and score['compound'] < 0.65:
      return "positif"
  elif abs(score['compound']) <= 0.25:
      return "neutre"
  elif score['compound'] < -0.25 and score['compound'] > -0.65:
      return "négatif"
  elif score['compound'] <= -0.6:
      return "très négatif"


### Début application ###

# Initialisation

st.markdown(
    """
<style>
span[data-baseweb="tag"] {
  background-color: orange !important;
}
</style>
""",
    unsafe_allow_html=True,
)


st.markdown("#  <center> :partly_sunny: Bilan des retours usagers :thunder_cloud_and_rain: </center> ", unsafe_allow_html=True)  

st.markdown("<h2> <b> Initialisation </b> </h2>", unsafe_allow_html=True)  



Debut = st.date_input(
    "Début de période",
    datetime.date(2023, 1, 1))

Fin = st.date_input(
    "Fin de période",
    datetime.date(2023, 7, 15))

Mot_cle = st.text_input('Filtrage avec un mot clé :', '')

st.write("Sélection des axes d'analyse des retours")
Statistiques_descriptives = st.checkbox('Statistiques descriptives')
Cartographie= st.checkbox('Cartographie')
Thematiques = st.checkbox('Thématiques')
Sentiments = st.checkbox('Sentiments  ( 30s / 100 lignes )')


# Base de données

df_retours = pd.read_csv("Data/Retours_usagers_reseaux_sociaux.csv", sep=";")




df_retours["Date"] = pd.to_datetime(df_retours["Date"], format="%d/%m/%Y")
date_debut = pd.to_datetime(str(Debut))
date_fin = pd.to_datetime(str(Fin))
df_retours = df_retours[df_retours["Date"] > date_debut]
df_retours = df_retours[df_retours["Date"] < date_fin]
df_retours = df_retours[df_retours['Contenu_retour'].str.contains(Mot_cle, case=False)]


texte="Ci dessous la base de données des retours usagers composé de "+str(df_retours.shape[0])+ " lignes."
st.write(texte)

st.dataframe(df_retours[["Date","Contenu_retour"]], hide_index=True,use_container_width=True)

# Analyse

if Statistiques_descriptives :
    st.markdown("<h2> <b> Statistiques descriptives </b> </h2>", unsafe_allow_html=True)  
    st.write("Jour le plus marquant :")
    jour_le_plus_marquant = df_retours['Date'].value_counts().idxmax()
    jour_le_plus_marquant = formater_date(jour_le_plus_marquant)
    st.write(jour_le_plus_marquant)
    st.write("Nuage de mot :")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    exclure_mots = ["ça","avant","-","min","haut","alors","rien","ligne","mois","TERHDFusagers","eu","quil","avoir","trains","après","aucun","prendre","bonjour","merci","va","depuis","suis","soit","faut","train","nest","très","ma","être","hautsdefrance","TERHDF","vers","peut","votre","faire","nous","même","entre","cela","fois","_","bien","quand","fait","donc","tout","tous","sans","mon","car","jai","si","y","mais","moi","pas","cest","avec","gare","TER","SNCF","object","Name","Texte","t","me","mais","cette","dtype","NaN","on","je",'d', 'du', 'de', 'la', 'des', 'le', 'et', 'est',"sest","été", 'elle', 'une',"non","son","dun","ne","ont", 'en', 'que', 'aux', 'qui', 'ces', 'les', 'dans', 'sur', 'l', 'un', 'pour', 'par', 'il', 'ou', 'à', 'ce', 'a', 'sont', 'cas', 'plus', 'leur', 'se', 's', 'vous', 'au', 'c', 'aussi', 'toutes', 'autre', 'comme']
    try :
      #wordcloud = WordCloud(background_color = 'white',width=1200, height=500, stopwords = exclure_mots, max_words = 50).generate(str(df_retours["Contenu_retour"].apply(lambda x: clean_text(str(x["Contenu_retour"])).tolist())))
      wordcloud = WordCloud(background_color='white', width=1200, height=500, stopwords=exclure_mots, max_words=38).generate(" ".join(df_retours["Contenu_retour"].apply(lambda x: clean_text(str(x))).tolist()))
      plt.imshow(wordcloud, interpolation='bilinear')
      plt.axis("off")
      plt.show()
      st.pyplot()
    except :
      wordcloud = WordCloud(background_color='white', width=1200, height=500, stopwords=exclure_mots, max_words=10).generate(" ".join(df_retours["Contenu_retour"].apply(lambda x: clean_text(str(x))).tolist()))
      plt.imshow(wordcloud, interpolation='bilinear')
      plt.axis("off")
      plt.show()
      st.pyplot()
      

    

if Cartographie :
    st.markdown("<h2> <b> Cartographie </b> </h2>", unsafe_allow_html=True)  
    st.markdown("Récupération des lieux géographiques dans les retours textuels des usagers et affichage sur une carte.")

    df_texte=df_retours[["Contenu_retour"]]
  
    df_texte.loc[:, "Contenu_retour"] = df_texte["Contenu_retour"].str.replace("PNO", "Paris")
    df_texte.loc[:, "Contenu_retour"] = df_texte["Contenu_retour"].str.replace("-", " et ")
    
    df_texte['Villes']=df_texte["Contenu_retour"].apply(lambda x: Extraction_ville(x))
    
    df_texte['Coordonnees']=df_texte["Villes"].apply(lambda x: Coordonnees_geographiques(x,df_villes))
    
    liste_ville=[]
    liste_latitude=[]
    liste_longitude=[]
    for i in df_texte["Coordonnees"]:
        for j in i :
            liste_ville.append(j["label"].values[0])
            liste_latitude.append(j["latitude"].values[0])
            liste_longitude.append(j["longitude"].values[0])
    df_carte=pd.DataFrame({'Nom_ville': liste_ville,'latitude': liste_latitude,'longitude': liste_longitude})
    df_carte['size'] = df_carte.groupby('Nom_ville')['Nom_ville'].transform('size')
    df_carte = df_carte.drop_duplicates(subset='Nom_ville')
    df_carte=df_carte[df_carte['size']>round(df_retours.shape[0]*0.01+1)]

    fig = px.scatter_mapbox(
        df_carte,
        lat="latitude",
        lon="longitude",
        hover_name="Nom_ville",
        zoom=5,
        mapbox_style="carto-positron",
        size="size"
    )
    fig.update_layout(
      title="Carte des retours usagers (>1%+1 occurences)",
      title_font_size=14,
      title_font_family="Arial"
    )
    st.plotly_chart(fig)

    Affichage_dataset= st.checkbox('Afficher le jeu de donnée')
    if Affichage_dataset :
      st.dataframe(df_texte[['Contenu_retour','Villes']], hide_index=True,use_container_width=True) # , hide_index=True,use_container_width=True

if Thematiques :
    
    exclure_mots = ["nord","etc","coucou","sous","quel","quoi","moins","terhdf","vraiment","quelle","terhdfusagers","où","pu","ça","dire","autres","doit","avant","__","-","min","haut","alors","rien","ligne","mois","TERHDFusagers","eu","quil","avoir","trains","après","aucun","prendre","bonjour","merci","va","depuis","suis","soit","faut","train","nest","très","ma","être","hautsdefrance","TERHDF","vers","peut","votre","faire","nous","même","entre","cela","fois","_","bien","quand","fait","donc","tout","tous","sans","mon","car","jai","si","y","mais","moi","pas","cest","avec","gare","TER","SNCF","object","Name","Texte","t","me","mais","cette","dtype","NaN","on","je",'d', 'du', 'de', 'la', 'des', 'le', 'et', 'est',"sest","été", 'elle', 'une',"non","son","dun","ne","ont", 'en', 'que', 'aux', 'qui', 'ces', 'les', 'dans', 'sur', 'l', 'un', 'pour', 'par', 'il', 'ou', 'à', 'ce', 'a', 'sont', 'cas', 'plus', 'leur', 'se', 's', 'vous', 'au', 'c', 'aussi', 'toutes', 'autre', 'comme']
    st.markdown("<h2> <b> Analyse des thématiques  </b> </h2>", unsafe_allow_html=True)  
    st.markdown("<b> Clusters : </b> ", unsafe_allow_html=True)
    df_retours.loc[:, "Contenu_retour"] = df_retours["Contenu_retour"].str.replace("PNO", "Paris")
    df_retours.loc[:, "Contenu_retour"] = df_retours["Contenu_retour"].str.replace("-", " et ")
    liste_df = list(df_retours.Contenu_retour)
    texts = liste_df
    texts=df_retours.Contenu_retour
    num_clusters = max(round(df_retours.shape[0]/100),2)
    max_features = 10000
    max_df = 0.8
    min_df = 2
    french_stop_words = list(set(stopwords.words('french')+["sncf","ter"]+exclure_mots))
    vectorizer = TfidfVectorizer(max_features=max_features, max_df=max_df, min_df=min_df, stop_words=french_stop_words)
    text_vect = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(text_vect)
    for i, cluster in enumerate(kmeans.cluster_centers_):
      st.markdown(f"Cluster {i} mots majoritaires :")
      #sorted_terms = [vectorizer.get_feature_names_out()[term_index] for term_index in cluster.argsort()[:-16 - 1:-1]]
      sorted_terms = [vectorizer.get_feature_names_out()[term_index] + f" ({cluster[term_index]:.2f})" for term_index in cluster.argsort()[:-16 - 1:-1]]
      st.markdown(', '.join(sorted_terms))
      labels = kmeans.labels_
      unique, counts = np.unique(labels, return_counts=True)
    st.markdown("<b> Effectifs : </b> ", unsafe_allow_html=True)  
    st.markdown(dict(zip(unique, counts)))
    # Calculer les pourcentages
    percentages = (counts / sum(counts)) * 100
    # Créer un dictionnaire avec les pourcentages
    percentage_dict = dict(zip(unique, percentages))
    for theme, percentage in percentage_dict.items():
      st.markdown(f"Thème {theme}: {percentage:.2f}%")

if Sentiments :
    st.markdown("<h2> <b> Analyse des sentiments </b> </h2>", unsafe_allow_html=True)  
    st.markdown("Répartition des sentiments")
    df_retours['Score']=df_retours['Contenu_retour'].apply(lambda x: translate_and_analyze_sentiment(x))
    df_retours['Score_categorie'] = df_retours['Score'].apply(lambda x: categorize_sentiment(x))
    df_retours['Score']=df_retours['Score'].apply(lambda x: x["compound"])
    df_retours["Comptage"]=1

    fig = px.pie(df_retours, values='Comptage', names='Score_categorie', color='Score_categorie',
             color_discrete_map={'très positif':'darkgreen',
                                 'positif':'green',
                                 'neutre':'grey',
                                 'négatif':'red',
                                 'très négatif':'darkred'})
    st.plotly_chart(fig)
  

    st.dataframe(df_retours[['Contenu_retour','Score_categorie',"Score"]], hide_index=True,use_container_width=True) # , hide_index=True,use_container_width=True



