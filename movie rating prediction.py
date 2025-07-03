import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import process
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title=" Movie Rating Predictor", layout="centered")
import base64
from pathlib import Path
#background image setup(bg saved in folder)
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
image_path = "C:/Users/Shraddha Saxena/OneDrive/Desktop/internship tasks/bg.jpg" 
bg_image = get_base64_image(image_path)
st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}
</style>
""", unsafe_allow_html=True)

#css for the page 
st.markdown("""
<style>
body {
    background-color: #121212;
    color: #E0E0E0;
    font-family: 'Poppins', sans-serif;
}
.title-text {
    font-size: 2.7rem;
    font-weight: 700;
    text-align: center;
    color: #72e0d1;
    margin-bottom: 30px;
    animation: fadeIn 1.2s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: scale(0.95); }
    to { opacity: 1; transform: scale(1); }
}
.metric-box {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 30px;
}
.recommend-title {
    font-size: 1.8rem;
    margin-top: 3rem;
    color: #72e0d1;
    text-align: center;
}
.card {
    background: linear-gradient(135deg, #1d1d1d, #232323);
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 12px;
    box-shadow: 0 0 10px rgba(114,224,209,0.2);
    color: #fff;
}
.rating-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
  margin-top: 30px;
}
.circular-chart {
  width: 140px;
  height: 140px;
  transform: rotate(-90deg);
}
.circle-bg {
  fill: none;
  stroke: #292929;
  stroke-width: 3.8;
}
.circle {
  fill: none;
  stroke-width: 3.8;
  stroke-linecap: round;
  transition: stroke-dasharray 1s ease-out;
  filter: drop-shadow(0 0 6px #72e0d1);
}
@keyframes fadeInCard {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.card {
    animation: fadeInCard 0.8s ease forwards;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    transform: scale(1.03);
    box-shadow: 0 0 15px rgba(114, 224, 209, 0.4);
    border: 1px solid #72e0d1;
}</style>
""", unsafe_allow_html=True)
st.markdown("<div class='title-text'>🎬 Movie Rating Predictor & Recommender</div>", unsafe_allow_html=True)
#taking dataset from the csv file 
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\\Users\\Shraddha Saxena\\OneDrive\\Desktop\\internship tasks\\IMDb Movies India.csv", encoding="ISO-8859-1")
    actor_cols = ['Actor 1', 'Actor 2', 'Actor 3']
    df['Genre'] = df['Genre'].fillna('')
    for col in actor_cols:
        df[col] = df[col].fillna('Unknown')
    df['num_genres'] = df['Genre'].apply(lambda x: len(x.split(',')))
    all_actors = pd.concat([df[col] for col in actor_cols])
    actor_counts = all_actors.value_counts().to_dict()
    df['actor_popularity'] = df.apply(lambda row: np.mean([actor_counts.get(row[col], 0) for col in actor_cols]), axis=1)
    top_directors = ['Christopher Nolan', 'Sanjay Leela Bhansali', 'Rajkumar Hirani', 'Anurag Kashyap']
    df['Director'] = df['Director'].fillna('Unknown')
    df['is_award_director'] = df['Director'].apply(lambda x: 1 if x in top_directors else 0)
    df['Duration'] = df['Duration'].astype(str).str.extract(r'(\d+)').astype(float)
    df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Rating', 'Duration', 'Year'])
    return df
df = load_data()
#clearing data and trainning model
@st.cache_resource
def train_model(df):
    X = df[['num_genres', 'actor_popularity', 'is_award_director', 'Duration', 'Year']]
    y = df['Rating']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model
model = train_model(df)
#user input's movie name 
movie_titles = df['Name'].dropna().unique()
movie_input = st.text_input("🔎 Search a Movie Title")
if movie_input:
    best_match, score = process.extractOne(movie_input, movie_titles)
    st.markdown(f"### Matched Movie: **{best_match}**")

    selected_row = df[df['Name'] == best_match].iloc[0]
    feature_row = pd.DataFrame([[selected_row['num_genres'],
                                 selected_row['actor_popularity'],
                                 selected_row['is_award_director'],
                                 selected_row['Duration'],
                                 selected_row['Year']]],
                               columns=['num_genres', 'actor_popularity', 'is_award_director', 'Duration', 'Year'])
    pred_rating = model.predict(feature_row)[0]
    rating_percent = min(max(pred_rating * 10, 0), 100)
    if pred_rating < 5.0:
        stroke_color = "#FF4C4C" 
    elif pred_rating < 6.0:
        stroke_color = "#FFD700"  
    else:
        stroke_color = "#12ba20"  
    #i set the color according to the rating we can change it according to our prediction standards

    st.markdown(f"""
    <div class="rating-wrapper">
      <div style="position: relative; display: inline-block;">
        <svg viewBox="0 0 36 36" class="circular-chart">
          <path class="circle-bg"
                d="M18 2.0845
                   a 15.9155 15.9155 0 0 1 0 31.831
                   a 15.9155 15.9155 0 0 1 0 -31.831"/>
          <path class="circle"
                stroke="{stroke_color}"
                stroke-dasharray="{rating_percent}, 100"
                d="M18 2.0845
                   a 15.9155 15.9155 0 0 1 0 31.831
                   a 15.9155 15.9155 0 0 1 0 -31.831"/>
        </svg>
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                    display: flex; align-items: center; justify-content: center;
                    font-size: 1.8rem; font-weight: bold; color: white; text-shadow: 0 0 4px #000;">
          <span>{pred_rating:.1f}</span>
          <span style="margin-left: 6px; font-size: 1.4rem;">🌟</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-top: 1rem; font-size: 0.95rem; color: #BBBBBB;">
        📊 <em>Rating predicted based on:</em><br>
        <strong>Genre overlap, actor popularity, director recognition, year & duration</strong>.
    </div>
    """, unsafe_allow_html=True)
    #calling for recommendation logic
    def get_similarity_score(row):
        genre_match = len(set(row['Genre'].split(',')) & set(selected_row['Genre'].split(',')))
        director_match = row['Director'] == selected_row['Director']
        actor_match = len(set([row['Actor 1'], row['Actor 2'], row['Actor 3']]) &
                          set([selected_row['Actor 1'], selected_row['Actor 2'], selected_row['Actor 3']]))
        return genre_match + actor_match + (1 if director_match else 0)
    df['similarity'] = df.apply(get_similarity_score, axis=1)
    similar_movies = df[df['Name'] != best_match].sort_values(by='similarity', ascending=False).head(3)
    st.markdown(f"<div class='recommend-title'>🎥 Recommended Movies Similar to <em>{best_match}</em></div>", unsafe_allow_html=True)
    for _, row in similar_movies.iterrows():
     st.markdown(f"""
        <div class="card">
            <h4> ⟫⟫{row['Name']} <span style='color: #90EE90;'>({int(row['Year'])})</span></h4><br>
            <p><strong>⭐ IMDb Rating:</strong> {row['Rating']}</p>
            <p><strong>🎭 Genre:</strong> {row['Genre']}</p>
            <p><strong>🎬 Director:</strong> {row['Director']}</p>
        </div>
    """, unsafe_allow_html=True)

