from flask import Flask, redirect, url_for, session, request, render_template, jsonify
import pyrebase
import os
from dotenv import load_dotenv
from datetime import datetime
import google.generativeai as genai

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

if __name__ == '__main__':
    app.run(debug=True)