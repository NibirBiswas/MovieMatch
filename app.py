import streamlit as st
import pickle
import pandas as pd
import requests
import os

def fetch_poster(id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=b22cb560447294856a2dba69bc7b10a0&language=en-US".format(id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        id = movies.iloc[i[0]].id
        recommended_movie_posters.append(fetch_poster(id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters

st.header('MovieMatch - a Project by Nibir Biswas')
movies_dict = pickle.load(open('movies.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

# Define the URL and local file path for 'similarity.pkl'
similarity_url = "URL_OF_THE_FILE"
similarity_file_path = "similarity.pkl"

def download_file(url, file_path):
    response = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(response.content)

# Download the file if it doesn't exist locally
if not os.path.exists(similarity_file_path):
    download_file(similarity_url, similarity_file_path)

# Load the 'similarity.pkl' file
similarity = pickle.load(open(similarity_file_path, 'rb'))

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "I like the movie below! (Type Movie Name)",
    movie_list
)
if st.button('Recommend Me!'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])
