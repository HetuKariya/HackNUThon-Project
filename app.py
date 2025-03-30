from flask import Flask, redirect, url_for, session, request, render_template, jsonify
import pyrebase
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime
import google.generativeai as genai
import joblib
import numpy as np

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-pro')

firebase_config = {
    "apiKey": os.getenv('FIREBASE_API_KEY'),
    "authDomain": os.getenv('FIREBASE_AUTH_DOMAIN'),
    "projectId": os.getenv('FIREBASE_PROJECT_ID'),
    "storageBucket": os.getenv('FIREBASE_STORAGE_BUCKET'),
    "messagingSenderId": os.getenv('FIREBASE_MESSAGING_SENDER_ID'),
    "appId": os.getenv('FIREBASE_APP_ID'),
    "databaseURL": os.getenv('FIREBASE_DATABASE_URL')
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            account_info = auth.get_account_info(user['idToken'])
            user_id = account_info['users'][0]['localId']

            session['user_id'] = user_id
            session['user_email'] = email
            session['logged_in'] = True
            session['id_token'] = user['idToken']

            user_data = db.child("users").child(user_id).get(user['idToken']).val()
            if user_data:
                session['user_name'] = user_data.get('name', 'User')

            return redirect(url_for('dashboard'))
        except Exception as e:
            error_message = "Invalid credentials. Please try again."
            return render_template('login.html', error=error_message)

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        try:
            user = auth.create_user_with_email_and_password(email, password)
            user_id = auth.get_account_info(user['idToken'])['users'][0]['localId']

            user_data = {
                "name": name,
                "email": email,
                "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            }
            db.child("users").child(user_id).set(user_data, user['idToken'])

            session['user_id'] = user_id
            session['user_name'] = name
            session['user_email'] = email
            session['logged_in'] = True
            session['id_token'] = user['idToken']

            return redirect(url_for('dashboard'))
        except Exception as e:
            error_code = e.args[1]['error']['message']
            if error_code == 'EMAIL_EXISTS':
                error_message = "Email already exists. Please use a different email."
            else:
                error_message = "An error occurred during signup. Please try again."
            return render_template('signup.html', error=error_message)

    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    return redirect(url_for('login'))

def refresh_token():
    user = auth.refresh(session['refresh_token'])
    session['id_token'] = user['idToken']
    return user['idToken']

@app.route('/gemini_chat', methods=['POST'])
def gemini_chat():
    if 'logged_in' not in session:
        return jsonify({'response': 'Please log in to use the chatbot.'}), 401
    data = request.json
    user_message = data.get('message', '')
    try:
        system_prompt = """You are an agricultural AI assistant for NutriSoil, specializing in providing 
        advice on soil health, crop management, and sustainable farming practices. 
        Provide helpful, accurate information with a friendly tone."""

        response = model.generate_content([
            {"role": "user", "parts": [system_prompt]},
            {"role": "model", "parts": ["I understand. I'll help with agricultural questions in a friendly way."]},
            {"role": "user", "parts": [user_message]}
        ])

        if 'user_id' in session:
            chat_data = {
                "user_id": session['user_id'],
                "user_message": user_message,
                "assistant_response": response.text,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            }
            db.child("chat_logs").push(chat_data, session.get('id_token'))

        return jsonify({'response': response.text})
    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({'response': 'Sorry, I encountered an error. Please try again.'}), 500

@app.route('/chatbot')
def chatbot():
    if 'logged_in' not in session:
        flash('Please log in to use the chatbot.', 'warning')
        return redirect(url_for('login'))
    return render_template('chatbot.html')

def load_models():
    try:
        model_soil = joblib.load('models/soil_type_model.pkl')
        model_growth = joblib.load('models/crop_growth_model.pkl')
        le_soil = joblib.load('models/soil_type_encoder.pkl')
        le_growth = joblib.load('models/crop_growth_encoder.pkl')
        return model_soil, model_growth, le_soil, le_growth
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None

model_soil, model_growth, le_soil, le_growth = load_models()

def categorize_soil(pH):
    """Soil categorization function"""
    if pd.isna(pH):
        return 'Unknown'
    elif pH < 4.5:
        return 'Very Acidic'
    elif 4.5 <= pH < 5.5:
        return 'Moderately Acidic'
    elif 5.5 <= pH < 6.5:
        return 'Slightly Acidic'
    elif 6.5 <= pH < 7.5:
        return 'Neutral'
    elif 7.5 <= pH < 8.5:
        return 'Mildly Alkaline'
    elif 8.5 <= pH < 9.5:
        return 'Moderately Alkaline'
    else:
        return 'Highly Alkaline'

def categorize_growth(moisture, ec, temp, nitrogen, phosphorus, potassium):
    score = 0

    if 25 <= moisture <= 35:
        score += 40
    elif (20 <= moisture < 25) or (35 < moisture <= 45):
        score += 30
    elif (15 <= moisture < 20) or (45 < moisture <= 55):
        score += 20
    elif (10 <= moisture < 15) or (55 < moisture <= 65):
        score += 10

    if 1 <= ec <= 3:
        score += 15
    elif 0.5 <= ec < 1 or 3 < ec <= 4:
        score += 10
    elif 0 <= ec < 0.5 or 4 < ec <= 5:
        score += 5

    if 18 <= temp <= 28:
        score += 15
    elif (15 <= temp < 18) or (28 < temp <= 32):
        score += 10
    elif (10 <= temp < 15) or (32 < temp <= 38):
        score += 5

    if nitrogen >= 30:
        score += 10
    elif 20 <= nitrogen < 30:
        score += 7
    elif 10 <= nitrogen < 20:
        score += 4

    if phosphorus >= 25:
        score += 10
    elif 15 <= phosphorus < 25:
        score += 7
    elif 5 <= phosphorus < 15:
        score += 4

    if potassium >= 25:
        score += 10
    elif 15 <= potassium < 25:
        score += 7
    elif 5 <= potassium < 15:
        score += 4

    if score >= 75:
        return "Optimal Growth"
    elif 50 <= score < 75:
        return "Good Growth"
    elif 30 <= score < 50:
        return "Moderate Growth"
    else:
        return "Poor Growth"

def recommend_fertilizer(soil_type, n_level, p_level, k_level):
    base_recommendations = {
        "Very Acidic": "Add agricultural lime (2-3 kg per 10 square meters) to raise pH.",
        "Moderately Acidic": "Apply dolomite lime (1-2 kg per 10 square meters) to gradually raise pH.",
        "Slightly Acidic": "Use light application of lime (0.5-1 kg per 10 square meters) if growing alkaline-loving plants.",
        "Neutral": "Maintain with compost for overall soil health.",
        "Mildly Alkaline": "Incorporate acidic organic matter like pine needles or peat moss.",
        "Moderately Alkaline": "Add elemental sulfur (100-200g per 10 square meters) to lower pH.",
        "Highly Alkaline": "Apply aluminum sulfate or iron sulfate (follow package instructions) for faster pH reduction."
    }

    npk_recommendations = []

    if n_level < 10:
        npk_recommendations.append(
            "Low Nitrogen: Apply high-nitrogen fertilizer like blood meal, fish emulsion, or urea.")
    elif 10 <= n_level < 20:
        npk_recommendations.append("Medium Nitrogen: Apply balanced NPK fertilizer.")

    if p_level < 10:
        npk_recommendations.append("Low Phosphorus: Apply rock phosphate, bone meal, or phosphate fertilizers.")
    elif 10 <= p_level < 20:
        npk_recommendations.append("Medium Phosphorus: Consider light application of phosphorus-containing fertilizer.")

    if k_level < 10:
        npk_recommendations.append("Low Potassium: Apply wood ash, greensand, or potassium sulfate.")
    elif 10 <= k_level < 20:
        npk_recommendations.append("Medium Potassium: Consider light application of potassium-containing fertilizer.")

    base_rec = base_recommendations.get(soil_type, "No specific soil pH recommendation.")
    npk_rec = " ".join(npk_recommendations)

    if npk_rec:
        return f"{base_rec}\n{npk_rec}"
    else:
        return f"{base_rec}\nNutrient levels (NPK) appear sufficient."

@app.route('/soil-analysis')
def soil_analysis():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    models_loaded = all([model_soil, model_growth, le_soil, le_growth])
    return render_template('soil_analysis.html', user=session, models_loaded=models_loaded)

@app.route('/analyze-soil', methods=['POST'])
def analyze_soil():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    try:
        ph = float(request.form.get('ph', 0))
        nitro = float(request.form.get('nitro', 0))
        posh_nitro = float(request.form.get('posh_nitro', 0))
        pota_nitro = float(request.form.get('pota_nitro', 0))
        moist = float(request.form.get('moist', 0))
        ec = float(request.form.get('ec', 0))
        temp = float(request.form.get('temp', 0))

        ph = max(0, min(14, ph))
        nitro = max(0, nitro)
        posh_nitro = max(0, posh_nitro)
        pota_nitro = max(0, pota_nitro)
        moist = max(0, moist)
        ec = max(0, ec)
        temp = max(-10, min(60, temp))

        if not all([model_soil, model_growth, le_soil, le_growth]):
            manual_soil_type = categorize_soil(ph)
            manual_growth_condition = categorize_growth(moist, ec, temp, nitro, posh_nitro, pota_nitro)
            fertilizer_advice = recommend_fertilizer(manual_soil_type, nitro, posh_nitro, pota_nitro)

            results = {
                'model_predictions': False,
                'soil_type': manual_soil_type,
                'growth_condition': manual_growth_condition,
                'fertilizer_advice': fertilizer_advice,
                'ph': ph,
                'nitro': nitro,
                'posh_nitro': posh_nitro,
                'pota_nitro': pota_nitro,
                'moist': moist,
                'ec': ec,
                'temp': temp
            }
        else:
            manual_soil_type = categorize_soil(ph)
            manual_growth_condition = categorize_growth(moist, ec, temp, nitro, posh_nitro, pota_nitro)
            fertilizer_advice = recommend_fertilizer(manual_soil_type, nitro, posh_nitro, pota_nitro)

            input_data_soil = np.array([[ph, nitro, posh_nitro, pota_nitro]])
            if hasattr(model_soil, 'named_steps') and 'scaler' in model_soil.named_steps:
                prediction_soil_idx = model_soil.predict(input_data_soil)[0]
            else:
                prediction_soil_idx = model_soil.predict(input_data_soil)[0]
            predicted_soil_type = le_soil.inverse_transform([prediction_soil_idx])[0]

            input_data_growth = np.array([[moist, ec, temp, nitro, posh_nitro, pota_nitro]])
            if hasattr(model_growth, 'named_steps') and 'scaler' in model_growth.named_steps:
                prediction_growth_idx = model_growth.predict(input_data_growth)[0]
            else:
                prediction_growth_idx = model_growth.predict(input_data_growth)[0]
            predicted_growth_condition = le_growth.inverse_transform([prediction_growth_idx])[0]

            additional_advice = []
            if ph < 5.5:
                additional_advice.append("Consider limestone application to raise soil pH")
                additional_advice.append("Choose acid-tolerant crops like blueberries, potatoes, or rhododendrons")
            elif ph > 7.5:
                additional_advice.append("Consider sulfur application to lower soil pH")
                additional_advice.append("Choose alkaline-tolerant crops like asparagus, beets, or cabbage")

            if nitro < 20:
                additional_advice.append("Low nitrogen may cause yellowing leaves and stunted growth")
            if posh_nitro < 20:
                additional_advice.append("Low phosphorus may cause poor root development and delayed flowering")
            if pota_nitro < 20:
                additional_advice.append("Low potassium may reduce disease resistance and fruit quality")

            if moist < 20:
                additional_advice.append("Consider irrigation systems to improve soil moisture")
            elif moist > 40:
                additional_advice.append("Consider improving drainage to reduce excess moisture")

            results = {
                'model_predictions': True,
                'ml_soil_type': predicted_soil_type,
                'ml_growth_condition': predicted_growth_condition,
                'soil_type': manual_soil_type,
                'growth_condition': manual_growth_condition,
                'fertilizer_advice': fertilizer_advice,
                'additional_advice': additional_advice,
                'ph': ph,
                'nitro': nitro,
                'posh_nitro': posh_nitro,
                'pota_nitro': pota_nitro,
                'moist': moist,
                'ec': ec,
                'temp': temp
            }

            if 'user_id' in session and 'id_token' in session:
                analysis_data = {
                    "user_id": session['user_id'],
                    "ph": ph,
                    "nitro": nitro,
                    "posh_nitro": posh_nitro,
                    "pota_nitro": pota_nitro,
                    "moist": moist,
                    "ec": ec,
                    "temp": temp,
                    "soil_type": manual_soil_type,
                    "growth_condition": manual_growth_condition,
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                }

                db.child("soil_analyses").push(analysis_data, session.get('id_token'))
        return render_template('soil_results.html', user=session, results=results)

    except Exception as e:
        print(f"Error in soil analysis: {e}")
        return render_template('soil_analysis.html', user=session, error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)