from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sleep Quality Prediction API!"}


# Ensure model file exists
MODEL_PATH = "sleep_quality_model_1.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please check the file path.")

# Load trained model
MODEL_TYPE = "pkl"  # Change to "h5" if using TensorFlow/Keras
model = joblib.load(MODEL_PATH)


# Define input schema
class UserInput(BaseModel):
    sleep_hours: float
    age: int
    gender: str
    caffeine_intake: int
    physical_activity: int
    stress_level: int


# Function to generate sleep & exercise recommendations
def get_recommendations(sleep_score):
    if 0 <= sleep_score <= 2:
        return {
            "sleep": [
                "Severely poor sleep. Aim for at least 7-9 hours of sleep.",
                "Avoid caffeine, nicotine, and alcohol at least 4-6 hours before bedtime.",
                "Reduce screen time at least 1-2 hours before bed to avoid blue light exposure.",
                "Establish a relaxing bedtime routine (reading, warm bath, or meditation).",
                "Ensure a dark, quiet, and cool sleep environment."
            ],
            "exercise": [
                "Gentle yoga or meditation to relax the mind before bed.",
                "Breathing exercises like 4-7-8 technique to improve sleep quality.",
                "Light stretching before bed to relieve tension and promote relaxation.",
                "Short walks in the evening (but avoid high-intensity workouts at night)."
            ]
        }

    elif 3 <= sleep_score <= 5:
        return {
            "sleep": [
                "Poor sleep. Maintain a consistent sleep schedule, even on weekends.",
                "Limit naps to 20-30 minutes during the day to avoid disrupting nighttime sleep.",
                "Use blackout curtains and a comfortable mattress to enhance sleep quality.",
                "Try herbal teas like chamomile or valerian root to promote relaxation.",
                "Reduce stress through relaxation techniques like journaling or gratitude practice."
            ],
            "exercise": [
                "Light aerobic exercises like walking, cycling, or swimming for 30 minutes daily.",
                "Tai Chi or gentle yoga to improve flexibility and relaxation.",
                "Morning or afternoon workouts (avoid intense exercises close to bedtime)."
            ]
        }

    elif 6 <= sleep_score <= 8:
        return {
            "sleep": [
                "Moderate sleep quality. Stick to a regular sleep schedule.",
                "Avoid heavy meals, spicy foods, and large amounts of liquid before bed.",
                "Ensure your bedroom is free from noise disruptions (use white noise machines if needed).",
                "Try progressive muscle relaxation to ease tension before sleeping.",
                "Expose yourself to natural sunlight in the morning to regulate circadian rhythm."
            ],
            "exercise": [
                "Moderate-intensity cardio like jogging, cycling, or brisk walking (150 mins/week).",
                "Strength training (2-3 times per week) to support overall health and sleep.",
                "Pilates or yoga for core strength and relaxation.",
                "Evening stretching routine to relax muscles before bed."
            ]
        }

    elif 9 <= sleep_score <= 10:
        return {
            "sleep": [
                "Excellent sleep! Keep following your healthy sleep habits.",
                "Maintain a balanced diet with proper hydration throughout the day.",
                "Continue avoiding screens before bedtime to maintain good sleep patterns.",
                "Incorporate mindfulness meditation for overall well-being.",
                "Consider tracking sleep patterns to identify further improvements."
            ],
            "exercise": [
                "High-intensity interval training (HIIT) for cardiovascular health (2-3 times/week).",
                "Strength training (weightlifting or resistance exercises) for muscle maintenance.",
                "Active hobbies like hiking, dancing, or swimming to stay fit and engaged.",
                "Consistent morning workouts to maintain energy levels and improve sleep."
            ]
        }

    else:
        return {
            "sleep": ["Invalid score. Please enter a score between 0 and 10."],
            "exercise": ["No recommendations available."]
        }


@app.post("/predict")
async def predict(input_data: UserInput):
    # Convert gender to numerical value (0 = Male, 1 = Female, -1 = Invalid)
    gender_mapping = {"male": 0, "female": 1}
    gender_value = gender_mapping.get(input_data.gender.lower(), -1)

    if gender_value == -1:
        return {"error": "Invalid gender input. Use 'male' or 'female'."}

    # Prepare input features
    user_features = np.array([[input_data.sleep_hours, input_data.age, gender_value,
                               input_data.caffeine_intake, input_data.physical_activity,
                               input_data.stress_level]])

    # Predict sleep quality score
    try:
        sleep_quality_score = model.predict(user_features)[0]
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    # Get sleep and exercise suggestions
    recommendations = get_recommendations(sleep_quality_score)

    return {
        "sleep_quality_score": float(sleep_quality_score),
        "recommendations": recommendations
    }
