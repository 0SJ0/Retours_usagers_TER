import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from mtranslate import translate

pd.set_option('display.max_colwidth', None)

nltk.download('vader_lexicon')

st.markdown("#  <center> :disappointed: Analyse des sentiments :blush:</center> ", unsafe_allow_html=True)  

st.markdown("Une analyse des sentiments peut donner une indication de la satisfaction générale des usagers. Ceci est essentiel pour évaluer si le service répond aux attentes des usagers et si des efforts supplémentaires doivent être déployés pour augmenter la satisfaction.")

st.markdown("En prenant en considération les opinions et les sentiments des usagers, une collectivité territoriale montre qu'elle est à l'écoute de ses citoyens et cela renforce la confiance entre les usagers et les autorités locales.")

st.markdown("Les collectivités territoriales mettent en place diverses politiques et initiatives pour améliorer les services de transport. Analyser les sentiments des usagers permet d'évaluer l'efficacité de ces politiques et d'ajuster l'offre en conséquence.")

st.markdown("<h2> <b> Concept : </b> </h2>", unsafe_allow_html=True)  

st.markdown("En mettant un texte dans l'espace en dessous, il est possible d'estimer la satisfaction de l'usager.")

def translate_french_to_english(text):
    return translate(text, 'en', 'fr')

def translate_french_to_english_reverse(text):
    return translate(text, 'fr', 'en')

# Initialisation de l'analyseur de sentiment
#sia = SentimentIntensityAnalyzer()


# Fonction pour traduire et calculer le score de sentiment
def translate_and_analyze_sentiment(text, source_lang='fr', target_lang='en'):
    sia = SentimentIntensityAnalyzer()
    # Traduction du texte
    translated_text = translate_french_to_english(text)
    #translated_text = text
    # Calcul du score de sentiment
    sentiment_score = sia.polarity_scores(translated_text)
    return sentiment_score


# Texte à analyser

texte = st.text_input("Retour usager :", "J'ai pris le 7h à Boulogne. C'est le meilleur trajet de ma vie. Merci beaucoup!")
temp=texte

#texte=translate_french_to_english(texte)

# Calcul du score de sentiment
#score = sia.polarity_scores(texte)
score = translate_and_analyze_sentiment(texte)

# Affichage du score
#print(score)


appreciation=""

# Catégorisation en fonction du score
if score['compound'] >= 0.4:
    appreciation="Le sentiment est très positif."
elif score['compound'] > 0:
    appreciation="Le sentiment est plutôt positif."
elif score['compound'] == 0:
    appreciation="Le sentiment est neutre."
elif score['compound'] > -0.4:
    appreciation="Le sentiment est plutôt négatif."
else:
    appreciation="Le sentiment est très négatif."
    
    
#print(score["compound"],appreciation)

noms = ["<b> Texte  =  </b>"+temp ,"<b>Score  =  </b>"+str(score['compound']),"<b>Appréciation  =  </b>"+ appreciation]

#st.write(texte)

df = pd.DataFrame(noms, columns=["Infos"])

for nom in noms:
    st.markdown(nom, unsafe_allow_html=True)
    
   
#st.markdown("[Exemple de retours des usagers](https://fr.trustpilot.com/review/www.ter-sncf.com)",unsafe_allow_html=True)

st.markdown("<h2> <b> Application : </b> </h2>", unsafe_allow_html=True)  

st.markdown("Essayons le concept sur une base test qui contient 25 tweets sur le hashtag #TERHdF.")

df_test=pd.read_csv("Data/Retours_usagers.csv", sep=';')
df_test=df_test[["Date","Contenu_retour","Source"]]

st.markdown("Résultat de l'analyse de sentiment (base et score) : ")

# Fonction pour calculer le score de sentiment d'une phrase
def calculate_sentiment_score(text):
    return translate_and_analyze_sentiment(text)['compound']

# Ajout de la colonne 'score' à df_test
df_test['score'] = df_test['Contenu_retour'].apply(calculate_sentiment_score)


st.dataframe(df_test[["Date","Contenu_retour","score"]])

#st.markdown("Graphique 1 (suivi temporel):")

#st.markdown("Graphique 3 (Répartition des sentiments, camembert):")

#st.markdown("Graphique 2 (Fréquence wordcloud):")





#st.dataframe(df)


