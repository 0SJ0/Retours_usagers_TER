[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sj-hdf-relation-usager-hdf-bienvenue-gce4r1.streamlit.app/)
[![Python](https://img.shields.io/badge/python-v3.8-blueviolet)](https://www.python.org/)
[![Plotly](https://img.shields.io/badge/plotly-5.10.0-blue)](https://plotly.com/)
[![NLTK](https://img.shields.io/badge/nltk-3.8.1-blue)](https://www.nltk.org/)
[![Scikit--learn](https://img.shields.io/badge/scikit--learn-1.3.0-blue)](https://scikit-learn.org/stable/)


# Relation usager HdF

Le code crée une application Streamlit pour visualiser et analyser les retours des usagers sur les réseaux sociaux concernant les transports. Il intègre des fonctionnalités pour filtrer les données par date et mot-clé, afficher des statistiques descriptives, créer une cartographie des retours basée sur les villes mentionnées, et effectuer une analyse thématique et sentimentale des textes. Les résultats sont présentés sous forme de tableaux, cartes et graphiques.

Lien du projet déployé : " https://sj-hdf-relation-usager-hdf-bienvenue-gce4r1.streamlit.app/ "

# Contenu du dépot

```
├── Data
│   ├── Retours_usagers.csv
│   ├── Retours_usagers_reseaux_sociaux.csv
│   ├── cities.csv
│   ├── cities_V2.csv
├── docs
│   ├── LICENSE.txt
├── pages
│   ├── Bilan_des_retours_.py
│   ├── Récupération_sectorielle.py
│   ├── Segmentation_retours.py
│   └── Sentiments_analyse.py
├── Bienvenue.py
├── README.md
├── fr_core_news_sm-3.5.0-py3-none-any.whl
├── requirements.txt
```

# Installation et exécution

Pour déployer cette application Streamlit en local :

**1 - Clonez ce repo sur votre machine :**

```
git clone https://path/Relation_usager_HdF.git
```

**2 - Naviguez jusqu'au dossier du repo :**

```
cd Relation_usager_HdF
```

**3 - (Optionnel) Créez un environnement virtuel pour isoler les dépendances :**

```
python -m venv venv
```

**4 - Activez l'environnement virtuel :**

Sur Windows :
```
.\venv\Scripts\activate
```
Sur macOS et Linux :
```
source venv/bin/activate
```
**5 - Installez les dépendances nécessaires :**

```
pip install -r requirements.txt
```
**6 - Lancez l'application Streamlit :**

```
streamlit run Bienvenue.py
```
**7 - Accédez à l'application dans un navigateur :**

L'application devrait être accessible à l'adresse suivante : http://localhost:8501

**8- Pour arrêter l'application :**

Appuyez sur Ctrl+C dans le terminal.

# Utilisation

L'application ouvre sur un menu. Il suffit de selectionner le projet recherché sur la barre de navigation.

# Dépendances


```
pandas >= 1.4
seaborn >= 0.11
streamlit >= 1.12.0
matplotlib >= 3.1
plotly >= 5.10.0
requests >= 2.31.0 
beautifulsoup4 >= 4.12.2 
mtranslate >= 1.8
nltk >= 3.8.1
geopy >= 2.4.0 
spacy >= 3.5.0
python-Levenshtein >= 0.21.1 
scikit-learn >= 1.3.0 
pydantic >= 1.8.2
wordcloud >= 1.9.2 
```


# Licence

GPL-3.0 license docs/LICENSE.txt



