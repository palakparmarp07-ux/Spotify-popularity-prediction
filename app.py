from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# loads the saved model and genre mapping
model = joblib.load('popularity_model.pkl')
genre_means = joblib.load('genre_means.pkl')

app = FastAPI(title="Spotify Popularity Predictor")

# defines what input data looks like
class SongFeatures(BaseModel):
    danceability: float
    energy: float
    loudness: float
    valence: float
    tempo: float
    acousticness: float
    speechiness: float
    genre: str

@app.get("/")
def home():
    return {"message": "Spotify Popularity Predictor API is running"}

@app.post("/predict")
def predict_popularity(song: SongFeatures):
    # encode the genre using the saved mapping; use overall average if genre is unseen
    genre_encoded = genre_means.get(song.genre, genre_means.mean())
    
    # builds the feature row in the exact order the model expects
    input_data = pd.DataFrame([{
        'danceability': song.danceability,
        'energy': song.energy,
        'loudness': song.loudness,
        'valence': song.valence,
        'tempo': song.tempo,
        'acousticness': song.acousticness,
        'speechiness': song.speechiness,
        'genre_encoded': genre_encoded
    }])
    
    prediction = model.predict(input_data)[0]
    
    return {
        "predicted_popularity": round(float(prediction), 2),
        "genre_used": song.genre,
        "genre_encoded_value": round(float(genre_encoded), 2)
    }