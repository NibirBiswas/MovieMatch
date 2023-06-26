import streamlit as st
import pickle
import pandas as pd
import requests
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to fetch movie poster
def fetch_poster(id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=b22cb560447294856a2dba69bc7b10a0&language=en-US".format(id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

# Function to convert text to list
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L

# Function to convert crew data
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 

# Load the preprocessed data and similarity matrix
movies_dict = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Create DataFrame from movies_dict
movies = pd.DataFrame(movies_dict)

# Apply data preprocessing steps
movies['cast'] = movies['cast'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: x[0:3])
movies['crew'] = movies['crew'].apply(fetch_director)
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
new = movies.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])

# Initialize CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new['tags']).toarray()

# Define recommend function
def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # Fetch the movie poster
        id = new.iloc[i[0]].id
        recommended_movie_posters.append(fetch_poster(id))
        recommended_movie_names.append(new.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters

# Streamlit app
def main():
    st.header('MovieMatch - a Project by Nibir Biswas')

    movie_list = new['title'].values
    selected_movie = st.selectbox("I like the movie below! (Type Movie Name)", movie_list)

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

if __name__ == '__main__':
    main()
