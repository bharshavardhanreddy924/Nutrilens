import os
import json
import numpy as np
import pandas as pd
import certifi
from datetime import datetime, timedelta
from collections import Counter

from flask import Flask, render_template, request, flash, redirect, url_for, session, jsonify
from flask_bcrypt import Bcrypt
from flask_cors import CORS

from keras.models import load_model
from PIL import Image, ImageOps
from pymongo import MongoClient
from langchain_groq import ChatGroq

# Environment Configuration
SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(24))
MONGODB_URI = os.environ.get('MONGODB_URI', 'mongodb+srv://bharshavardhanreddy924:uLCmWytTkthYz3xJ@data-dine.5oghq.mongodb.net/?retryWrites=true&w=majority&ssl=true')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', 'your_groq_api_key')
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flask App Initialization
app = Flask(__name__)
app.secret_key = SECRET_KEY
CORS(app)
bcrypt = Bcrypt(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# MongoDB Configuration
client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
db = client["NutriLens"]
food_logs_collection = db["food_logs"]
users_collection = db["users"]

# Constants
BOWL_SIZES = {
    "mini": {"name": "Mini Bowl", "multiplier": 1.0},
    "small": {"name": "Small Bowl", "multiplier": 2.0},
    "medium": {"name": "Medium Bowl", "multiplier": 3.5},
    "large": {"name": "Large Bowl", "multiplier": 5.0},
    "xl": {"name": "Extra-Large Bowl", "multiplier": 7.5},
    "jumbo": {"name": "Jumbo Bowl", "multiplier": 10.0},
}

DAILY_RECOMMENDATIONS = {
    "Energy (Kcal)": 2000,
    "Protein (g)": 60,
    "Fat (g)": 70,
    "Carbohydrates (g)": 310
}

# Load Model and Nutritional Data
try:
    model = load_model("working_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = [' '.join(name.strip().split()[1:]) for name in f.readlines()]
    nutritional_data = pd.read_csv("NutritionalValues.csv", index_col="Food Item")
except Exception as e:
    print(f"Error loading resources: {e}")
    model, class_names, nutritional_data = None, [], pd.DataFrame()

def load_recommendations():
    recommendations = {}
    nutrient_files = {
        "Energy (Kcal)": "energy.txt",
        "Protein (g)": "protein.txt",
        "Fat (g)": "fat.txt",
        "Carbohydrates (g)": "carbs.txt"
    }
    
    for nutrient, filename in nutrient_files.items():
        try:
            with open(filename, 'r') as f:
                recommendations[nutrient] = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            recommendations[nutrient] = []
            print(f"Warning: {filename} not found")
    
    return recommendations

def get_nutrient_recommendations(total_nutrition):
    recommendations = load_recommendations()
    nutrient_advice = {}
    
    nutrition_mapping = {
        "calories": "Energy (Kcal)",
        "protein": "Protein (g)",
        "carbohydrates": "Carbohydrates (g)",
        "fat": "Fat (g)"
    }
    
    for total_key, recommend_key in nutrition_mapping.items():
        current_value = total_nutrition.get(total_key, 0)
        recommended_value = DAILY_RECOMMENDATIONS[recommend_key]
        
        if current_value < recommended_value * 0.8:
            nutrient_advice[recommend_key] = {
                'current': round(current_value, 1),
                'recommended': recommended_value,
                'percentage': round((current_value / recommended_value) * 100, 1),
                'foods': recommendations.get(recommend_key, [])
            }
    
    return nutrient_advice

def adjust_nutritional_info(nutritional_info, multiplier):
    if nutritional_info is None:
        return None
    
    return {key: value * multiplier if isinstance(value, (int, float)) else value 
            for key, value in nutritional_info.items()}

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part in request', 'danger')
            return redirect(request.url)

        file = request.files['image']
        bowl_size = request.form.get('bowl_size', 'mini')

        if bowl_size not in BOWL_SIZES:
            flash('Invalid bowl size selected', 'danger')
            return redirect(request.url)

        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)

        if file:
            try:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)

                image = Image.open(file_path).convert("RGB")
                image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
                image_array = np.asarray(image)
                normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                data[0] = normalized_image_array

                prediction = model.predict(data)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence_score = float(prediction[0][index])

                nutritional_info = None
                if class_name in nutritional_data.index:
                    nutritional_info = nutritional_data.loc[class_name].to_dict()
                    multiplier = BOWL_SIZES[bowl_size]["multiplier"]
                    nutritional_info = adjust_nutritional_info(nutritional_info, multiplier)

                food_log = {
                    "username": session.get("username", "guest"),
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "food_item": class_name,
                    "bowl_size": BOWL_SIZES[bowl_size]["name"],
                    "portion_weight": f"{int(100 * BOWL_SIZES[bowl_size]['multiplier'])}g",
                    "confidence_score": confidence_score,
                    "nutritional_info": nutritional_info,
                    "image_path": file_path
                }
                food_logs_collection.insert_one(food_log)

                flash('File uploaded and processed successfully!', 'success')
                return render_template(
                    'index.html',
                    image_url=file_path,
                    class_name=class_name,
                    confidence_score=confidence_score,
                    nutritional_info=nutritional_info,
                    bowl_sizes=BOWL_SIZES,
                    selected_bowl=bowl_size
                )
            except Exception as e:
                flash(f'Error processing image: {e}', 'danger')
                return redirect(request.url)

    return render_template('index.html', bowl_sizes=BOWL_SIZES)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users_collection.find_one({"username": username})
        if user and bcrypt.check_password_hash(user['password'], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if users_collection.find_one({"username": username}):
            flash('Username already exists.', 'danger')
        else:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            users_collection.insert_one({"username": username, "password": hashed_password})
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

class IndianDietBot:
    def __init__(self, api_key: str):
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name="mixtral-8x7b-32768"
        )
        
    def respond_to_query(self, user_query: str) -> str:
        prompt = f"""Indian diet expert query:
        {user_query}
        
        Provide a concise, expert response focusing on nutritional insights."""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error processing query: {str(e)}"

diet_bot = IndianDietBot(GROQ_API_KEY)

@app.route('/chatbot')
def chatbot():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('chatbot.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    user_message = data.get('message')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        response = diet_bot.respond_to_query(user_message)
        return jsonify({
            'response': response,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
