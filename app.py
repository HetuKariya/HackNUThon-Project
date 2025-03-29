from flask import Flask, redirect, url_for, session, request, render_template, jsonify
import pyrebase
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

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

if __name__ == '__main__':
    app.run(debug=True)