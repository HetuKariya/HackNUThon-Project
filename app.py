from flask import Flask, redirect, url_for, session, request, render_template, jsonify, flash
import pyrebase
import pandas as pd
import os
import json
from dotenv import load_dotenv
from datetime import datetime
from groq import Groq
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'fallback-secret-key-change-in-production')

# Configure Groq client
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# Firebase config
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────── Auth Routes ───────────────────────

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
        except Exception:
            return render_template('login.html', error="Invalid credentials. Please try again.")

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
            error_str = str(e)

            if 'EMAIL_EXISTS' in error_str:
                error_message = "Email already exists. Please use a different email."
            elif 'WEAK_PASSWORD' in error_str:
                error_message = "Password is too weak. Please use at least 6 characters."
            else:
                error_message = "An error occurred during signup. Please try again."

            return render_template('signup.html', error=error_message)

    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
def index():
    return redirect(url_for('login'))


# ─────────────────────── Dashboard ───────────────────────

@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session)


# ─────────────────────── Chatbot ───────────────────────

@app.route('/chatbot')
def chatbot():
    if 'logged_in' not in session:
        flash('Please log in to use the chatbot.', 'warning')
        return redirect(url_for('login'))

    return render_template('chatbot.html')


@app.route('/gemini_chat', methods=['POST'])
def gemini_chat():
    if 'logged_in' not in session:
        return jsonify({'response': 'Please log in to use the chatbot.'}), 401

    data = request.json
    user_message = data.get('message', '')

    try:
        chat_completion = groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an agricultural AI assistant for NutriSoil, specializing in "
                        "providing advice on soil health, crop management, and sustainable farming "
                        "practices."
                    )
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            temperature=0.7,
            max_tokens=1024,
        )

        reply_text = chat_completion.choices[0].message.content

        if 'user_id' in session:
            chat_data = {
                "user_id": session['user_id'],
                "user_message": user_message,
                "assistant_response": reply_text,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            }

            try:
                db.child("chat_logs").push(chat_data, session.get('id_token'))
            except Exception:
                pass

        return jsonify({'response': reply_text})

    except Exception as e:
        print(e)
        return jsonify({'response': 'Error generating response'}), 500


# ─────────────────────── ML Models ───────────────────────

def load_models():
    try:
        model_soil = joblib.load(os.path.join(BASE_DIR, "models", "soil_type_model.pkl"))
        model_growth = joblib.load(os.path.join(BASE_DIR, "models", "crop_growth_model.pkl"))
        le_soil = joblib.load(os.path.join(BASE_DIR, "models", "soil_type_encoder.pkl"))
        le_growth = joblib.load(os.path.join(BASE_DIR, "models", "crop_growth_encoder.pkl"))
        return model_soil, model_growth, le_soil, le_growth
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None


model_soil, model_growth, le_soil, le_growth = load_models()


# ─────────────────────── Soil Analysis Routes ───────────────────────

@app.route('/soil-analysis')
def soil_analysis():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    models_loaded = all([model_soil, model_growth, le_soil, le_growth])

    return render_template(
        'soil_analysis.html',
        user=session,
        models_loaded=models_loaded
    )


# ─────────────────────── Gallery & Spectral ───────────────────────

@app.route('/gallery')
def gallery():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    return render_template('gallery.html', user=session)


@app.route('/spectral-analysis')
def spectral_analysis():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    model_results = []

    for model_num in range(1, 13):
        result = {
            'model_number': model_num,
            'filename': f'model_{model_num}.pkl',
            'metrics': {},
            'plot1_img': None,
            'plot2_img': None
        }

        model_path = os.path.join(BASE_DIR, "static", "models", f"model_{model_num}.pkl")

        if os.path.exists(model_path):
            result['file_size'] = round(os.path.getsize(model_path) / 1024, 2)

        model_results.append(result)

    return render_template(
        'spectral_results.html',
        user=session,
        model_results=model_results
    )


# ─────────────────────── Render Server Fix ───────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)