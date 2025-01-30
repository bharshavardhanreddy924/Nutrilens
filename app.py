import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from flask import Flask, render_template, request, flash, redirect, url_for, session, jsonify
from flask_bcrypt import Bcrypt
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
from langchain_groq import ChatGroq

app = Flask(__name__)
bcrypt = Bcrypt(app)

# Secret key for sessions and security
app.secret_key = os.urandom(24)

# Configure the upload folder
UPLOAD_FOLDER = 'uploads'
if os.path.exists(UPLOAD_FOLDER):
    if not os.path.isdir(UPLOAD_FOLDER):
        print(f"Error: '{UPLOAD_FOLDER}' exists but is not a directory.")
else:
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        print(f"Upload folder created at: {UPLOAD_FOLDER}")
    except Exception as e:
        print(f"Error creating upload folder: {e}")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
import certifi
# MongoDB Configuration
from pymongo.mongo_client import MongoClient
uri = "mongodb+srv://bharshavardhanreddy924:516474Ta@data-dine.5oghq.mongodb.net/?retryWrites=true&w=majority&ssl=true"
client = MongoClient(uri, tlsCAFile=certifi.where())

db = client["NutriLens"]
food_logs_collection = db["food_logs"]
users_collection = db["users"]

from flask_socketio import SocketIO, emit, join_room, leave_room
from bson.objectid import ObjectId
from datetime import datetime, timedelta
import json

# Initialize SocketIO with Flask app
socketio = SocketIO(app)

# Add these collections to MongoDB setup
friends_collection = db["friends"]
chat_collection = db["chats"]
friend_requests_collection = db["friend_requests"]

# Friend Request System
@app.route('/send_friend_request/<username>', methods=['POST'])
def send_friend_request(username):
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401
        
    if username == session['username']:
        return jsonify({"error": "Cannot send friend request to yourself"}), 400
        
    # Check if user exists
    target_user = users_collection.find_one({"username": username})
    if not target_user:
        return jsonify({"error": "User not found"}), 404
        
    # Check if already friends
    existing_friendship = friends_collection.find_one({
        "$or": [
            {"user1": session['username'], "user2": username},
            {"user1": username, "user2": session['username']}
        ]
    })
    
    if existing_friendship:
        return jsonify({"error": "Already friends"}), 400
        
    # Check if request already sent
    existing_request = friend_requests_collection.find_one({
        "from_user": session['username'],
        "to_user": username,
        "status": "pending"
    })
    
    if existing_request:
        return jsonify({"error": "Friend request already sent"}), 400
        
    # Create friend request
    friend_requests_collection.insert_one({
        "from_user": session['username'],
        "to_user": username,
        "status": "pending",
        "date": datetime.now()
    })
    
    return jsonify({"success": True, "message": "Friend request sent"})

@app.route('/handle_friend_request/<request_id>/<action>', methods=['POST'])
def handle_friend_request(request_id, action):
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401
        
    friend_request = friend_requests_collection.find_one({
        "_id": ObjectId(request_id),
        "to_user": session['username']
    })
    
    if not friend_request:
        return jsonify({"error": "Friend request not found"}), 404
        
    if action == "accept":
        # Create friendship
        friends_collection.insert_one({
            "user1": friend_request['from_user'],
            "user2": friend_request['to_user'],
            "date": datetime.now()
        })
        status = "accepted"
    elif action == "reject":
        status = "rejected"
    else:
        return jsonify({"error": "Invalid action"}), 400
        
    # Update request status
    friend_requests_collection.update_one(
        {"_id": ObjectId(request_id)},
        {"$set": {"status": status}}
    )
    
    return jsonify({"success": True, "message": f"Friend request {status}"})

# Search Users
@app.route('/search_users')
def search_users():
    if 'username' not in session:
        return redirect(url_for('login'))
        
    query = request.args.get('query', '')
    if len(query) < 3:
        return jsonify([])
        
    users = users_collection.find({
        "username": {"$regex": query, "$options": "i"},
        "username": {"$ne": session['username']}
    }, {"username": 1, "_id": 0}).limit(10)
    
    return jsonify([user['username'] for user in users])

# Chat System
@socketio.on('join')
def on_join(data):
    room = data['room']
    join_room(room)

@socketio.on('leave')
def on_leave(data):
    room = data['room']
    leave_room(room)

@socketio.on('send_message')
def handle_message(data):
    room = data['room']
    message = {
        "sender": session['username'],
        "content": data['message'],
        "timestamp": datetime.now(),
        "room": room
    }
    
    chat_collection.insert_one(message)
    emit('new_message', {
        "sender": message['sender'],
        "content": message['content'],
        "timestamp": message['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
    }, room=room)

# Friend Progress Tracking
@app.route('/friend/progress/<username>')
def friend_progress(username):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Check if friends
    friendship = friends_collection.find_one({
        "$or": [
            {"user1": session['username'], "user2": username},
            {"user1": username, "user2": session['username']}
        ]
    })
    
    if not friendship:
        flash("You must be friends to view their progress", "error")
        return redirect(url_for('index'))
    
    # Get both users' goals
    current_user_goals = users_collection.find_one({"username": session['username']})
    friend_goals = users_collection.find_one({"username": username})
    
    # Get logs for last 7 days for both users
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Initialize daily nutrients dict for both users
    def init_daily_nutrients():
        nutrients = {}
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            nutrients[date_str] = {
                "calories": 0,
                "protein": 0,
                "carbohydrates": 0,
                "fat": 0,
                "goal_achievement": 0  # Percentage of daily goals met
            }
            current_date += timedelta(days=1)
        return nutrients
    
    user_nutrients = init_daily_nutrients()
    friend_nutrients = init_daily_nutrients()
    
    # Get and process logs for both users
    def process_user_logs(username, nutrients_dict, goals):
        logs = list(food_logs_collection.find({
            "username": username,
            "date": {
                "$gte": start_date.strftime("%Y-%m-%d"),
                "$lte": end_date.strftime("%Y-%m-%d")
            }
        }).sort("date", -1))
        
        # Define daily goals
        daily_goals = {
            "calories": goals.get("daily_calories", 2000),
            "protein": goals.get("daily_protein", 50),
            "carbohydrates": goals.get("daily_carbs", 250),
            "fat": goals.get("daily_fat", 70)
        }
        
        for log in logs:
            date = log['date']
            if nutritional_info := log.get("nutritional_info"):
                nutrients_dict[date]["calories"] += nutritional_info.get("Energy (Kcal)", 0)
                nutrients_dict[date]["protein"] += nutritional_info.get("Protein (g)", 0)
                nutrients_dict[date]["carbohydrates"] += nutritional_info.get("Carbohydrates (g)", 0)
                nutrients_dict[date]["fat"] += nutritional_info.get("Fat (g)", 0)
                
                # Calculate goal achievement percentage
                daily_achievements = [
                    min(nutrients_dict[date]["calories"] / daily_goals["calories"] * 100, 100),
                    min(nutrients_dict[date]["protein"] / daily_goals["protein"] * 100, 100),
                    min(nutrients_dict[date]["carbohydrates"] / daily_goals["carbohydrates"] * 100, 100),
                    min(nutrients_dict[date]["fat"] / daily_goals["fat"] * 100, 100)
                ]
                nutrients_dict[date]["goal_achievement"] = sum(daily_achievements) / len(daily_achievements)
        
        return logs
    
    user_logs = process_user_logs(session['username'], user_nutrients, current_user_goals)
    friend_logs = process_user_logs(username, friend_nutrients, friend_goals)
    
    # Calculate overall achievement scores
    def calculate_achievement_stats(nutrients_dict):
        dates = nutrients_dict.keys()
        total_achievement = sum(nutrients_dict[date]["goal_achievement"] for date in dates)
        avg_achievement = total_achievement / len(dates) if dates else 0
        
        return {
            "average_achievement": round(avg_achievement, 1),
            "best_day": max(dates, key=lambda d: nutrients_dict[d]["goal_achievement"]) if dates else None,
            "streak": calculate_streak(nutrients_dict)
        }
    
    def calculate_streak(nutrients_dict):
        streak = 0
        dates = sorted(nutrients_dict.keys(), reverse=True)
        for date in dates:
            if nutrients_dict[date]["goal_achievement"] >= 80:  # Consider 80% achievement as a successful day
                streak += 1
            else:
                break
        return streak
    
    user_stats = calculate_achievement_stats(user_nutrients)
    friend_stats = calculate_achievement_stats(friend_nutrients)
    
    return render_template(
        'friend_progress.html',
        friend_username=username,
        daily_nutrients=friend_nutrients,
        logs=friend_logs,
        user_nutrients=user_nutrients,
        user_logs=user_logs,
        user_stats=user_stats,
        friend_stats=friend_stats,
        current_user_goals=current_user_goals,
        friend_goals=friend_goals
    )
# Chat Interface
@app.route('/chat/<username>')
def chat_interface(username):
    if 'username' not in session:
        return redirect(url_for('login'))
        
    # Check if friends
    friendship = friends_collection.find_one({
        "$or": [
            {"user1": session['username'], "user2": username},
            {"user1": username, "user2": session['username']}
        ]
    })
    
    if not friendship:
        flash("You must be friends to chat", "error")
        return redirect(url_for('index'))
        
    # Generate unique room ID
    users = sorted([session['username'], username])
    room = f"chat_{users[0]}_{users[1]}"
    
    # Get chat history
    messages = chat_collection.find({
        "room": room
    }).sort("timestamp", 1)
    
    return render_template(
        'chat.html',
        friend_username=username,
        room=room,
        messages=messages
    )

# Friends List
@app.route('/friends')
def friends_list():
    if 'username' not in session:
        return redirect(url_for('login'))
        
    # Get all friends - convert cursor to list
    friends = friends_collection.find({
        "$or": [
            {"user1": session['username']},
            {"user2": session['username']}
        ]
    })
    
    friend_usernames = []
    for friend in friends:
        friend_username = friend['user1'] if friend['user1'] != session['username'] else friend['user2']
        friend_usernames.append(friend_username)
    
    # Get pending friend requests - convert cursor to list
    pending_requests = list(friend_requests_collection.find({
        "to_user": session['username'],
        "status": "pending"
    }))
    
    return render_template(
        'friends.html',
        friends=friend_usernames,
        pending_requests=pending_requests
    )


BOWL_SIZES = {
    "mini": {"name": "Mini Bowl", "multiplier": 1.0},  # 100g
    "small": {"name": "Small Bowl", "multiplier": 2.0},  # 200g
    "medium": {"name": "Medium Bowl", "multiplier": 3.5},  # 350g
    "large": {"name": "Large Bowl", "multiplier": 5.0},  # 500g
    "xl": {"name": "Extra-Large Bowl", "multiplier": 7.5},  # 750g
    "jumbo": {"name": "Jumbo Bowl", "multiplier": 10.0},  # 1000g
}

# Define recommended daily values
DAILY_RECOMMENDATIONS = {
    "Energy (Kcal)": 2000,
    "Protein (g)": 60,
    "Fat (g)": 70,
    "Carbohydrates (g)": 310
}

# Load the pre-trained model and class labels
try:
    model = load_model("working_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = [' '.join(name.strip().split()[1:]) for name in f.readlines()]
except Exception as e:
    print(f"Error loading model or labels: {e}")
    class_names = []
    model = None

# Load the nutritional data from a CSV file
try:
    nutritional_data = pd.read_csv("NutritionalValues.csv", index_col="Food Item")
except Exception as e:
    print(f"Error loading nutritional data: {e}")
    nutritional_data = pd.DataFrame()

def load_recommendations():
    """Load food recommendations from text files"""
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
    """
    Determine which nutrients are below recommended values and return appropriate food recommendations
    """
    recommendations = load_recommendations()
    nutrient_advice = {}
    
    # Map the total_nutrition keys to their corresponding DAILY_RECOMMENDATIONS keys
    nutrition_mapping = {
        "calories": "Energy (Kcal)",
        "protein": "Protein (g)",
        "carbohydrates": "Carbohydrates (g)",
        "fat": "Fat (g)"
    }
    
    for total_key, recommend_key in nutrition_mapping.items():
        current_value = total_nutrition.get(total_key, 0)
        recommended_value = DAILY_RECOMMENDATIONS[recommend_key]
        
        if current_value < recommended_value * 0.8:  # Below 80% of recommended value
            nutrient_advice[recommend_key] = {
                'current': round(current_value, 1),
                'recommended': recommended_value,
                'percentage': round((current_value / recommended_value) * 100, 1),
                'foods': recommendations.get(recommend_key, [])
            }
    
    return nutrient_advice

def adjust_nutritional_info(nutritional_info, multiplier):
    """Adjust nutritional information based on bowl size multiplier"""
    if nutritional_info is None:
        return None
    
    adjusted_info = {}
    for key, value in nutritional_info.items():
        if isinstance(value, (int, float)):
            adjusted_info[key] = value * multiplier
        else:
            adjusted_info[key] = value
    return adjusted_info

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Check if an image file is uploaded
        if 'image' not in request.files:
            flash('No file part in request', 'danger')
            return redirect(request.url)

        file = request.files['image']
        bowl_size = request.form.get('bowl_size', 'mini')  # Default to 'mini' bowl size

        # Validate bowl size
        if bowl_size not in BOWL_SIZES:
            flash('Invalid bowl size selected', 'danger')
            return redirect(request.url)

        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)

        if file:
            try:
                # Save the file
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)

                # Process the image and predict
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

                # Fetch nutritional info and adjust based on bowl size
                nutritional_info = None
                if class_name in nutritional_data.index:
                    nutritional_info = nutritional_data.loc[class_name].to_dict()
                    multiplier = BOWL_SIZES[bowl_size]["multiplier"]
                    nutritional_info = adjust_nutritional_info(nutritional_info, multiplier)

                # Log the food item to the database
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

@app.route('/history', methods=['GET'])
def history():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session.get("username", "guest")
    date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    logs = list(food_logs_collection.find({"username": username, "date": date}))

    total_nutrition = {
        "calories": 0,
        "protein": 0,
        "carbohydrates": 0,
        "fat": 0
    }

    for log in logs:
        log["_id"] = str(log["_id"])
        if log.get("nutritional_info"):
            total_nutrition["calories"] += log["nutritional_info"].get("Energy (Kcal)", 0)
            total_nutrition["protein"] += log["nutritional_info"].get("Protein (g)", 0)
            total_nutrition["carbohydrates"] += log["nutritional_info"].get("Carbohydrates (g)", 0)
            total_nutrition["fat"] += log["nutritional_info"].get("Fat (g)", 0)
    
    # Get nutritional recommendations based on current intake
    nutrient_recommendations = get_nutrient_recommendations(total_nutrition)

    return render_template(
        "history.html",
        logs=logs,
        date=date,
        total_nutrition=total_nutrition,
        recommendations=nutrient_recommendations,
        daily_recommendations=DAILY_RECOMMENDATIONS
    )

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
# Add these imports at the top of your file
from datetime import datetime, timedelta
from collections import Counter
import json

# Add this new route to your Flask application
@app.route('/weekly_analysis')
def weekly_analysis():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session.get("username")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Fetch last 7 days of food logs
    logs = list(food_logs_collection.find({
        "username": username,
        "date": {
            "$gte": start_date.strftime("%Y-%m-%d"),
            "$lte": end_date.strftime("%Y-%m-%d")
        }
    }))
    
    # Initialize data structures for analysis
    daily_nutrients = {
        (start_date + timedelta(days=x)).strftime("%Y-%m-%d"): {
            "calories": 0,
            "protein": 0,
            "carbohydrates": 0,
            "fat": 0
        } for x in range(8)
    }
    
    food_frequency = Counter()
    total_calories = 0
    meal_counts = 0
    
    # Process logs
    for log in logs:
        date = log["date"]
        food_item = log["food_item"]
        food_frequency[food_item] += 1
        
        if nutritional_info := log.get("nutritional_info"):
            daily_nutrients[date]["calories"] += nutritional_info.get("Energy (Kcal)", 0)
            daily_nutrients[date]["protein"] += nutritional_info.get("Protein (g)", 0)
            daily_nutrients[date]["carbohydrates"] += nutritional_info.get("Carbohydrates (g)", 0)
            daily_nutrients[date]["fat"] += nutritional_info.get("Fat (g)", 0)
            total_calories += nutritional_info.get("Energy (Kcal)", 0)
            meal_counts += 1
    
    # Calculate statistics
    avg_daily_calories = total_calories / 7 if meal_counts > 0 else 0
    favorite_foods = food_frequency.most_common(5)
    
    # Prepare data for charts
    nutrient_trends = {
        "dates": list(daily_nutrients.keys()),
        "calories": [data["calories"] for data in daily_nutrients.values()],
        "protein": [data["protein"] for data in daily_nutrients.values()],
        "carbohydrates": [data["carbohydrates"] for data in daily_nutrients.values()],
        "fat": [data["fat"] for data in daily_nutrients.values()]
    }
    
    # Calculate completion percentages
    weekly_totals = {
        "calories": sum(data["calories"] for data in daily_nutrients.values()),
        "protein": sum(data["protein"] for data in daily_nutrients.values()),
        "carbohydrates": sum(data["carbohydrates"] for data in daily_nutrients.values()),
        "fat": sum(data["fat"] for data in daily_nutrients.values())
    }
    
    weekly_goals = {
        "calories": DAILY_RECOMMENDATIONS["Energy (Kcal)"] * 7,
        "protein": DAILY_RECOMMENDATIONS["Protein (g)"] * 7,
        "carbohydrates": DAILY_RECOMMENDATIONS["Carbohydrates (g)"] * 7,
        "fat": DAILY_RECOMMENDATIONS["Fat (g)"] * 7
    }
    
    completion_percentages = {
        nutrient: (total / weekly_goals[nutrient] * 100) 
        for nutrient, total in weekly_totals.items()
    }
    
    return render_template(
        "weekly_analysis.html",
        nutrient_trends=json.dumps(nutrient_trends),
        favorite_foods=favorite_foods,
        avg_daily_calories=round(avg_daily_calories, 1),
        meal_counts=meal_counts,
        completion_percentages=completion_percentages,
        weekly_totals=weekly_totals,
        weekly_goals=weekly_goals
    )
from recom import DietRecommender   
@app.route('/recommend')
def recommend_page():
    return render_template('recommend.html')  # Ensure 'recommend.html' is created in the templates folder

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    try:
        age = int(data['age'])
        veg_status = int(data['veg_status'])
        weight = float(data['weight'])
        height = float(data['height'])
        
        recommendations = DietRecommender.get_recommendations(age, veg_status, weight, height)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


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
        prompt = f"""
        You are an Indian diet expert. Answer the following query concisely:
        
        {user_query}
        
        If the query is about meal recommendations, include:
        1. Dish name
        2. Main ingredients
        3. Approximate nutritional information
        
        If the query is unrelated to Indian diets, please provide a general response stating that I am only well-versed as a diet recommender.
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"

# Initialize the chatbot with API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_api_key_here")
diet_bot = IndianDietBot(GROQ_API_KEY)

# Add these new routes to your Flask application
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
