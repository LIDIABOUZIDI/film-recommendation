import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("🎬 Système de recommandation de films")

# Chargement des données
def load_data():
    try:
        df = pd.read_csv("user_ratings_genres_mov.csv")
        st.success("Données chargées avec succès")
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {str(e)}")
        return None

df = load_data()

# Récupération des genres uniques
def get_all_genres(df):
    genres = set()
    for genre_list in df['genres'].dropna():
        for genre in genre_list.split('|'):
            genres.add(genre)
    return sorted(genres)

# Interface utilisateur
if df is not None:
    st.header("🔍 Sélectionnez vos préférences de films")
    preferences = []
    all_genres = get_all_genres(df)
    film_choices = df['title'].unique().tolist()

    with st.form("preferences_form"):
        selected_films = []
        for i in range(3):
            available_films = [film for film in film_choices if film not in selected_films]
            film_name = st.selectbox(f"Nom du film {i+1}", options=available_films, key=f"name_{i}")

            # Sélection des genres multiples
            selected_genres = st.multiselect(f"Genres pour le film {i+1}", options=all_genres, key=f"genre_{i}")
            genre_combination = "|".join(selected_genres)

            film_rating = st.slider(f"Note (1-5) pour le film {i+1}", 1, 5, 3, key=f"rating_{i}")
            preferences.append((film_name, genre_combination, film_rating))
            selected_films.append(film_name)

        rec_type = st.selectbox("Type de recommandation", ["NMF", "SVD", "KNN", "Contenu"], key="rec_type")
        submit = st.form_submit_button("💾 Valider")

    if submit:
        if all(film for film, genre, rating in preferences):
            # Mettre à jour les données
            for film, genre, rating in preferences:
                new_row = {'userId': 'user_999', 'title': film, 'rating': rating, 'genres': genre}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            pivot = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)
            user_index = pivot.index.get_loc('user_999')

            # Recommandation
            if rec_type == "NMF":
                model = NMF(n_components=10, init='random', random_state=0)
                user_features = model.fit_transform(pivot)
                item_features = model.components_
                scores = np.dot(user_features[user_index], item_features)
            elif rec_type == "SVD":
                model = TruncatedSVD(n_components=10)
                user_features = model.fit_transform(pivot)
                item_features = model.components_
                scores = np.dot(user_features[user_index], item_features)
            elif rec_type == "KNN":
                model = NearestNeighbors(metric='cosine', algorithm='brute')
                model.fit(pivot)
                distances, indices = model.kneighbors([pivot.iloc[user_index]], n_neighbors=6)
                similar_users = indices.flatten()[1:]
                scores = pivot.iloc[similar_users].mean().values
            else:
                tfidf = TfidfVectorizer(tokenizer=lambda x: x.split('|'))
                tfidf_matrix = tfidf.fit_transform(df['genres'].fillna(""))
                best_film = max(preferences, key=lambda x: x[2])[0]
                film_index = df[df['title'] == best_film].index[0]
                cosine_sim = cosine_similarity(tfidf_matrix)
                similar_scores = list(enumerate(cosine_sim[film_index]))
                similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)

                recommendations = []
                seen_titles = set()
                for idx, score in similar_scores:
                    film_title = df['title'][idx]
                    if film_title not in seen_titles and film_title not in selected_films:
                        recommendations.append((film_title, round(score, 4)))
                        seen_titles.add(film_title)
                        if len(recommendations) >= 5:
                            break
                recommended_movies = pd.DataFrame(recommendations, columns=['title', 'Score'])

            # Traitement des résultats pour NMF, SVD et KNN
            if rec_type in ["NMF", "SVD", "KNN"]:
                recommended_movies = pd.DataFrame({'title': pivot.columns, 'Score': scores})
                recommended_movies = recommended_movies[~recommended_movies['title'].isin(selected_films)]
                recommended_movies = recommended_movies.sort_values(by='Score', ascending=False).head(5)

            # Enlever les doublons et normaliser les scores
            recommended_movies = recommended_movies.drop_duplicates(subset=['title'])
            recommended_movies['Score'] = recommended_movies['Score'].clip(lower=0).round(4)

            st.header(f"🎯 Recommandations - {rec_type}")
            st.table(recommended_movies)
        else:
            st.warning("Veuillez remplir tous les champs avant de valider.")
else:
    st.error("Impossible de charger les données.")
