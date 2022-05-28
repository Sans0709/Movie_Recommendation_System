import streamlit as st
from PIL import Image
import pandas as pd

def run(movie_id,userId,r,cat):
 import numpy as np
 import pandas as pd
 import sklearn
 import matplotlib.pyplot as plt
 import seaborn as sns
 from sklearn.neighbors import NearestNeighbors
 from collections import Counter
 from scipy.sparse import csr_matrix

 import warnings
 warnings.simplefilter(action='ignore', category=FutureWarning)

 ratings = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/ratings.csv")

 movies = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")

 no_of_ratings = len(ratings)
 no_of_movies = len(ratings['movieId'].unique())
 no_of_users = len(ratings['userId'].unique())

 movies['genres'] = movies['genres'].apply(lambda x: x.split("|"))

 genre_frequency = Counter(g for genres in movies['genres'] for g in genres)

 def create_X(df):
     """
     Generates a sparse matrix from ratings dataframe.

      Args:
         df: pandas dataframe containing 3 columns (userId, movieId, rating)

     Returns:
         X: sparse matrix
         user_mapper: dict that maps user id's to user indices
         user_inv_mapper: dict that maps user indices to user id's
         movie_mapper: dict that maps movie id's to movie indices
         movie_inv_mapper: dict that maps movie indices to movie id's
     """
     M = df['userId'].nunique()
     N = df['movieId'].nunique()

     user_mapper = dict(zip(np.unique(df["userId"]), list(range(M))))
     movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(N))))

     user_inv_mapper = dict(zip(list(range(M)), np.unique(df["userId"])))
     movie_inv_mapper = dict(zip(list(range(N)), np.unique(df["movieId"])))

     user_index = [user_mapper[i] for i in df['userId']]
     item_index = [movie_mapper[i] for i in df['movieId']]

     X = csr_matrix((df["rating"], (user_index,item_index)), shape=(M,N))

     return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

 X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)

 n_total = X.shape[0]*X.shape[1]
 n_ratings = X.nnz
 sparsity = n_ratings/n_total
 n_ratings_per_movie = X.getnnz(axis=0)

 sum_ratings_per_movie = X.sum(axis=0)
 mean_rating_per_movie = sum_ratings_per_movie/n_ratings_per_movie

 X_mean_movie = np.tile(mean_rating_per_movie, (X.shape[0],1))
 X_mean_movie.shape
 X_norm = X - csr_matrix(X_mean_movie)

 def find_similar_movies(movie_id, X, movie_mapper, movie_inv_mapper, k, metric='cosine'):
     """
     Finds k-nearest neighbours for a given movie id.

     Args:
         movie_id: id of the movie of interest
         X: user-item utility matrix
         k: number of similar movies to retrieve
         metric: distance metric for kNN calculations

     Output: returns list of k similar movie ID's
     """
     X = X.T
     neighbour_ids = []

     movie_ind = movie_mapper[movie_id]
     movie_vec = X[movie_ind]
     if isinstance(movie_vec, (np.ndarray)):
         movie_vec = movie_vec.reshape(1,-1)
     # use k+1 since kNN output includes the movieId of interest
     kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
     kNN.fit(X)
     neighbour = kNN.kneighbors(movie_vec, return_distance=False)
     for i in range(0,k):
         n = neighbour.item(i)
         neighbour_ids.append(movie_inv_mapper[n])
     neighbour_ids.pop(0)
     return neighbour_ids  
 movie_titles = dict(zip(movies['movieId'], movies['title']))
 similar_movies = find_similar_movies(movie_id, X_norm, movie_mapper, movie_inv_mapper, metric='cosine', k=r)
 movie_title = movie_titles[movie_id]
 n_movies = movies['movieId'].nunique()
 genres = set(g for G in movies['genres'] for g in G)
 for g in genres:
   movies[g] = movies.genres.transform(lambda x: int(g in x))
 movie_genres = movies.drop(columns=['movieId', 'title','genres'])

 from sklearn.decomposition import TruncatedSVD
 svd = TruncatedSVD(n_components=20, n_iter=10)
 Z = svd.fit_transform(X.T)
 similar_movies = find_similar_movies(movie_id, Z.T, movie_mapper, movie_inv_mapper, metric='cosine', k=r)
 movie_title = movie_titles[movie_id]

 if cat==1 :
   st.write("Your N recommendation ")
   for i in similar_movies:
     st.write(movie_titles[i])     

 elif cat==2 :
  new_X = svd.inverse_transform(Z).T
  user_preferences = ratings[(ratings['userId']==userId)&(ratings['rating']>=4)]
  user_preferences = user_preferences.merge(movies[['movieId', 'title']])

  movie_titles = dict(zip(movies['movieId'], movies['title']))
  top_N=r
  top_N_indices = new_X[user_mapper[userId]].argsort()[-top_N:][::-1] 
  st.write("Top Recommendations for UserId ")
  for i in top_N_indices:
    movie_id = movie_inv_mapper[i]
    st.write(movie_titles[movie_id])


movies = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")
img1= Image.open('/content/image.jpg')
img1=img1.resize((150,150),)
st.image(img1,use_column_width=False)
st.markdown("<h1 font-size:40px;color: white;'>MovieApp</h1>", unsafe_allow_html=True)
category = ['--Select--', 'Based on Movie', 'Based on userId']
cat_op = st.selectbox('Select Recommendation Type', category)
movie=movies['movieId']
if cat_op == category[0]:
  st.warning('Please select Recommendation Type!!')
elif cat_op == category[1]:
  select_movie = st.selectbox('Select movie: (Recommendation will be based on this selection)', movie)  
  movie_id=int(select_movie)
  movie_titles = dict(zip(movies['movieId'], movies['title']))
  movie_title = movie_titles[movie_id]
  st.write(movie_title)
  no_of_reco = st.slider('Number of movies you want Recommended:', min_value=5, max_value=20, step=1)
  run(movie_id,0,no_of_reco,1)
elif cat_op ==category[2]:
  user = st.number_input("Enter User Id",step=1)
  if(st.button('Submit')):
   no_of_reco = st.slider('Number of movies you want Recommended:', min_value=5, max_value=20, step=1) 
   st.write(user)
   run(0,user,no_of_reco,2)
   
  

