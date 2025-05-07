from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import io
import json
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import pipeline, AutoModelForImageClassification, AutoImageProcessor
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import uvicorn
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, Text,DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
import threading
from datetime import datetime
from sqlalchemy import func, extract
from json import loads
from PIL import Image
import os
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from PIL import Image
import io
import json

# --- API Keys and Configs ---
GEMINI_API_KEY = "AIzaSyCs84cKrL7QTIutVBHlmK4R9MIeP8wkrow"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

SPOTIFY_CLIENTS = [
    {
        "client_id": "8f8d8c8d3df04f7aaea05ec10964fd9b",
        "client_secret": "ce2e00baeb9945de97a935906d5ceaa9",
    },
    {
        "client_id": "90a9503c028c4868aa9423081e58e59b",
        "client_secret": "bc32eb23e6844c13b95e15c0ab24581d",
    }
]

DATABASE_URL = "postgresql://postgres:1234@localhost/emotion_db"

# --- Database setup ---
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserEmotion(Base):
    __tablename__ = "user_emotions"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True, nullable=False)  # Changed from user_id
    combined_text = Column(Text)
    text_emotion = Column(String)
    image_emotion = Column(String)
    final_emotion = Column(String)
    song_1_name = Column(String)
    song_1_artist = Column(String)
    song_1_link = Column(String)
    song_1_duration = Column(String)
    song_1_image = Column(String)
    song_2_name = Column(String)
    song_2_artist = Column(String)
    song_2_link = Column(String)
    song_2_duration = Column(String)
    song_2_image = Column(String)
    song_3_name = Column(String)
    song_3_artist = Column(String)
    song_3_link = Column(String)
    song_3_duration = Column(String)
    song_3_image = Column(String)
    song_4_name = Column(String)
    song_4_artist = Column(String)
    song_4_link = Column(String)
    song_4_duration = Column(String)
    song_4_image = Column(String)
    song_5_name = Column(String)
    song_5_artist = Column(String)
    song_5_link = Column(String)
    song_5_duration = Column(String)
    song_5_image = Column(String)
    playlist_name = Column(String)
    playlist_link = Column(String)
    playlist_image = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# --- Initialize models ---
text_classifier = pipeline("text-classification", model="michellejieli/emotion_text_classifier")
image_processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
image_model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")

# --- Spotify Client Management ---
spotipy.Spotify.__del__ = lambda self: None
spotipy.oauth2.SpotifyAuthBase.__del__ = lambda self: None

spotify_client_counter = 0
spotify_client_lock = threading.Lock()

def get_spotify_client():
    global spotify_client_counter
    with spotify_client_lock:
        creds = SPOTIFY_CLIENTS[spotify_client_counter % len(SPOTIFY_CLIENTS)]
        spotify_client_counter += 1
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=creds["client_id"],
        client_secret=creds["client_secret"]
    ))

# --- FastAPI setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    user_id: int
    text: str

# --- Utility Functions ---

def predict_image_emotion(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = image_processor(img, return_tensors="pt")
    with torch.no_grad():
        outputs = image_model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)
    idx = logits.argmax(-1).item()
    emotion = image_model.config.id2label[idx].lower()
    confidence = probabilities[0, idx].item()
    return emotion, round(confidence * 100, 2)

def format_duration(duration_ms):
    minutes = duration_ms // 60000
    seconds = (duration_ms % 60000) // 1000
    return f"{minutes}:{seconds:02d}"

def generate_spotify_search_query(emotion: str) -> str:
    prompt = f"""
You are a super creative music explorer.
Given the emotion '{emotion}', invent a random, wild, and highly creative Spotify search query.
Mix and match ideas like:
- Music decades (70s, 80s, 90s, early 2000s)
- Genres (lo-fi, indie, metal, trap, jazz, classical, etc.)
- Situations (late night drives, rainy mornings, heartbreak, dance party alone, road trips)
- Moods (moody, energetic, sleepy, dreamy)
- Weird ideas (songs for staring out of windows, crying while cooking, lost in the forest)
Make each search query unique, human-like, and very different from others.

**Only output the search query text, nothing else.**
Make it sound natural like a real Spotify search.

Example styles:
- "90s breakup anthems"
- "lofi chill beats for late night sadness"
- "songs to cry while watching rain"
- "indie soft tunes for lonely hearts"
- "early 2000s sad pop classics"
"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 1.2}
    )
    return response.text.strip()

# --- API Endpoint ---

@app.post("/predict/")
async def predict_all(email: str = Form(...), text: str = Form(...), image_path: str = Form(...)):
    db = SessionLocal()  # Initialize your DB session
    print("🔄 Database session started.")

    try:
        # Step 1: Handle image processing
        print(f"📂 Received image path: {image_path}")
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        SAVE_FOLDER = "predict_uploads"
        os.makedirs(SAVE_FOLDER, exist_ok=True)
        save_path = os.path.join(SAVE_FOLDER, os.path.basename(image_path))
        print(f"📁 Saving image to: {save_path}")

        with open(image_path, "rb") as src_file, open(save_path, "wb") as dst_file:
            image_bytes = src_file.read()
            dst_file.write(image_bytes)

        print("✅ Image successfully saved.")

        # Step 2: Load and process image
        with open(save_path, "rb") as f:
            processed_image_bytes = f.read()

        # Step 3: Predict emotion from image
        print("🔍 Predicting emotion from image...")
        img_emotion, img_confidence = predict_image_emotion(processed_image_bytes)
        print(f"🖼️ Image Emotion: {img_emotion} ({img_confidence}%)")

        # Step 4: Predict emotion from text
        print("🧠 Predicting emotion from text...")
        text_result = text_classifier(text)[0]
        text_emotion = text_result['label'].lower()
        text_confidence = round(text_result['score'] * 100, 2)
        print(f"📝 Text Emotion: {text_emotion} ({text_confidence}%)")

        # Step 5: Decide final emotion
        final_emotion = text_emotion if text_confidence >= img_confidence else img_emotion
        print(f"🎯 Final Emotion: {final_emotion}")

        # Step 6: Spotify Search for Songs
        query = generate_spotify_search_query(final_emotion)
        spotify = get_spotify_client()
        print(f"🎵 Searching Spotify for emotion: {final_emotion} → Query: {query}")
        search_results = spotify.search(q=query, type="track", limit=5)
        tracks = search_results.get('tracks', {}).get('items', [])

        # Step 7: Format songs
        songs = [{
            "name": t.get("name"),
            "artist": t["artists"][0].get("name"),
            "url": t.get("external_urls", {}).get("spotify"),
            "album_image": t.get("album", {}).get("images", [{}])[0].get("url"),
            "duration": format_duration(t.get("duration_ms", 0))
        } for t in tracks]
        print(f"🎧 Top 5 Songs Found: {[s['name'] for s in songs]}")

        # Step 8: Search for Playlist
        playlist_search = spotify.search(q=f"{query} playlist", type="playlist", limit=1)
        playlists = playlist_search.get('playlists', {}).get('items', [])
        playlist = None
        if playlists:
            p = playlists[0]
            playlist = {
                "name": p.get("name"),
                "url": p.get("external_urls", {}).get("spotify"),
                "image": p.get("images", [{}])[0].get("url")
            }
            print(f"📚 Playlist Found: {playlist['name']}")
        else:
            print("📚 No playlist found.")

        # Step 9: Save to database
        print("💾 Saving prediction result to the database...")
        entry = UserEmotion(
            email=email,
            combined_text=text,
            text_emotion=text_emotion,
            image_emotion=img_emotion,
            final_emotion=final_emotion,
            song_1_name=songs[0].get("name"),
            song_1_artist=songs[0].get("artist"),
            song_1_link=songs[0].get("url"),
            song_1_duration=songs[0].get("duration"),
            song_1_image=songs[0].get("album_image"),
            song_2_name=songs[1].get("name"),
            song_2_artist=songs[1].get("artist"),
            song_2_link=songs[1].get("url"),
            song_2_duration=songs[1].get("duration"),
            song_2_image=songs[1].get("album_image"),
            song_3_name=songs[2].get("name"),
            song_3_artist=songs[2].get("artist"),
            song_3_link=songs[2].get("url"),
            song_3_duration=songs[2].get("duration"),
            song_3_image=songs[2].get("album_image"),
            song_4_name=songs[3].get("name"),
            song_4_artist=songs[3].get("artist"),
            song_4_link=songs[3].get("url"),
            song_4_duration=songs[3].get("duration"),
            song_4_image=songs[3].get("album_image"),
            song_5_name=songs[4].get("name"),
            song_5_artist=songs[4].get("artist"),
            song_5_link=songs[4].get("url"),
            song_5_duration=songs[4].get("duration"),
            song_5_image=songs[4].get("album_image"),
            playlist_name=playlist.get("name") if playlist else None,
            playlist_link=playlist.get("url") if playlist else None,
            playlist_image=playlist.get("image") if playlist else None
        )

        db.add(entry)
        db.commit()
        db.refresh(entry)
        print("✅ Data saved to DB successfully.")

        return {
            "status": "success",
            "final_emotion": final_emotion,
            "songs": songs,
            "playlist": playlist
        }

    except Exception as e:
        db.rollback()
        print(f"❌ Error: {str(e)}")
        return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})
    finally:
        db.close()
        print("🔚 Database session closed.")

# # API to get the frequency of a particular emotion per user
# @app.get("/emotion_frequency/{email}/{emotion}")
# async def get_emotion_frequency(email: str, emotion: str):
#     db = SessionLocal()

#     # Count how many times the given emotion appears in the final_emotion column
#     emotion_count = db.query(func.count(UserEmotion.id)).filter(
#         UserEmotion.email == email,
#         UserEmotion.final_emotion == emotion
#     ).scalar()

#     return {"email": email, "emotion": emotion, "count": emotion_count}

# # API to get the weekday data (Monday = 0, Sunday = 6) for a particular user's emotions
# # Add weekday names map
# WEEKDAY_NAMES = {
#     0: "Monday",
#     1: "Tuesday",
#     2: "Wednesday",
#     3: "Thursday",
#     4: "Friday",
#     5: "Saturday",
#     6: "Sunday"
# }

# @app.get("/emotion_weekday_data/{email}")
# async def get_emotion_weekday_data(email: str):
#     db = SessionLocal()

#     try:
#         # Query grouped by weekday
#         emotion_weekday_counts = db.query(
#             extract('dow', UserEmotion.timestamp).label('weekday'),
#             func.count(UserEmotion.id).label('count')
#         ).filter(UserEmotion.email == email).group_by('weekday').all()

#         # Initialize with zero for all weekdays
#         weekday_counts = {day: 0 for day in WEEKDAY_NAMES.values()}

#         # Fill actual counts
#         for row in emotion_weekday_counts:
#             weekday_name = WEEKDAY_NAMES.get(row.weekday, "Unknown")
#             weekday_counts[weekday_name] = row.count

#         return {
#             "email": email,
#             "weekday_counts": weekday_counts
#         }

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

#     finally:
#         db.close()

# # API to calculate the total duration of songs played by a user@app.get("/total_song_duration/{email}")
# async def get_total_song_duration(email: str):
#     db = SessionLocal()
#     total_duration_ms = 0

#     try:
#         # Get user's emotion entries
#         user_emotions = db.query(UserEmotion).filter(UserEmotion.email == email).all()

#         for entry in user_emotions:
#             songs = loads(entry.songs)  # Deserialize the songs field
#             for song in songs:
#                 duration_str = song.get("duration", "0:00")  # Ensure the key exists
#                 try:
#                     # Parse the duration (MM:SS) format
#                     minutes, seconds = map(int, duration_str.split(":"))
#                     duration_ms = (minutes * 60 + seconds) * 1000  # Convert to milliseconds
#                     total_duration_ms += duration_ms
#                 except Exception:
#                     continue  # skip songs with bad format

#         # Convert total duration to minutes and seconds
#         total_duration_min = total_duration_ms // 60000
#         total_duration_sec = (total_duration_ms % 60000) // 1000

#         return {
#             "email": email,
#             "total_duration": f"{total_duration_min}:{total_duration_sec:02d}",
#             "total_duration_minutes": round(total_duration_ms / 60000, 2)
#         }

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

#     finally:
#         db.close()

# --- Main ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=9000,reload=True)