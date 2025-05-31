from flask import Flask, request, render_template, send_file, redirect, url_for, session, flash
import os
import pickle
import sqlite3
import pandas as pd
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from train_model import train_phinet_model
from functools import wraps
import smtplib
from email.message import EmailMessage
import random
import string
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load model and feature engine
if not os.path.exists('model.pkl') or not os.path.exists('feature_engine.pkl'):
    raise FileNotFoundError("Model files not found. Run train_model.py first.")

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('feature_engine.pkl', 'rb') as f:
    feature_engine = pickle.load(f)

# Initialize database
def init_db():
    conn = sqlite3.connect("email_analysis.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS emails (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Email_id TEXT,
        email_body TEXT,
        URL TEXT,
        attached_file TEXT,
        Label TEXT
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS access_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        username TEXT,
        action TEXT,
        resource TEXT,
        result TEXT
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT,
        department TEXT,
        verified BOOLEAN,
        clearance TEXT
    )''')
    conn.commit()
    conn.close()

# Helpers
def get_current_user():
    return session.get("user")

def log_access(user, action, resource, allowed):
    conn = sqlite3.connect("email_analysis.db")
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO access_logs (timestamp, username, action, resource, result)
                      VALUES (?, ?, ?, ?, ?)''',
                   (datetime.now().isoformat(), user['username'], action, resource, 'ALLOWED' if allowed else 'DENIED'))
    conn.commit()
    conn.close()

def check_access(user, action, resource):
    allowed = True
    if not user:
        return False
    if action == "retrain_model" and user.get("role") != "admin":
        allowed = False
    elif action == "view_results" and user.get("department") != "Security":
        allowed = False
    elif action == "submit_email" and not user.get("verified"):
        allowed = False
    elif action == "download_csv" and user.get("role") != "admin" and user.get("clearance") != "high":
        allowed = False
    log_access(user, action, resource, allowed)
    return allowed

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not get_current_user():
            return render_template('access_denied.html')
        return f(*args, **kwargs)
    return decorated_function

def generate_temp_password(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def send_reset_email(to_email, temp_password):
    sender_email = os.getenv("EMAIL_ADDRESS")
    sender_password = os.getenv("EMAIL_PASSWORD")

    msg = EmailMessage()
    msg['Subject'] = 'Password Reset - Phishing Detection App'
    msg['From'] = sender_email
    msg['To'] = to_email
    msg.set_content(f'Your temporary password is: {temp_password}\nPlease login and change it immediately.')

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
            smtp.starttls()
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
    except Exception as e:
        print("Email error:", e)

# Routes
@app.route('/')
def landing():
    return redirect(url_for('index')) if get_current_user() else render_template('landing.html')

@app.route('/home')
@login_required
def home():
    return render_template('index.html', user=get_current_user())

@app.route('/index')
@login_required
def index():
    return render_template('index.html', user=get_current_user())

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    user = get_current_user()
    if not check_access(user, "submit_email", "email_analysis"):
        return render_template('access_denied.html')

    email_id = request.form.get('emailId')
    email_body = request.form.get('emailBody', '')
    email_url = request.form.get('emailUrl', '')
    email_files = request.files.getlist('emailFiles')
    file_names = ' '.join([file.filename for file in email_files if file.filename])

    new_data = pd.DataFrame([{
        'email_id': email_id,
        'email_body': email_body,
        'urls': email_url,
        'attachments': file_names
    }])

    features = feature_engine.transform(new_data)
    prediction = model.predict(features)[0]
    phishing_result = 'Phishing Email' if prediction == 1 else 'Legitimate Email'

    conn = sqlite3.connect("email_analysis.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO emails (Email_id, email_body, URL, attached_file, Label) VALUES (?, ?, ?, ?, ?)",
                   (email_id, email_body, email_url, file_names, phishing_result))
    conn.commit()
    conn.close()

    return render_template('results.html', email_data={
        'emailId': email_id,
        'emailBody': email_body,
        'emailUrl': email_url,
        'emailFiles': file_names,
        'phishingResult': phishing_result
    }, user=user)

@app.route('/retrain', methods=['POST'])
@login_required
def retrain():
    user = get_current_user()
    if not check_access(user, "retrain_model", "model"):
        return render_template('access_denied.html')

    df = pd.read_csv('email_data.csv', on_bad_lines='skip')
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns={'url': 'urls', 'attached_file': 'attachments', 'label': 'label'}, inplace=True)
    df = df.dropna(subset=['email_body', 'label'])
    df = df[df['label'].isin(['Phishing Email', 'Legitimate Email'])]
    df['label'] = df['label'].map({'Phishing Email': 1, 'Legitimate Email': 0})

    if df['label'].nunique() < 2 or df.shape[0] < 5:
        return "Not enough data to retrain.", 400

    global model, feature_engine
    model, feature_engine = train_phinet_model(df)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('feature_engine.pkl', 'wb') as f:
        pickle.dump(feature_engine, f)

    return render_template('retrain_success.html', user=user)

@app.route('/download_csv')
@login_required
def download_csv():
    if not check_access(get_current_user(), "download_csv", "email_data"):
        return render_template('access_denied.html')
    return send_file("email_data.csv", as_attachment=True)

@app.route('/view_logs')
@login_required
def view_logs():
    user = get_current_user()
    if user.get("role") != "admin":
        return render_template('access_denied.html')

    conn = sqlite3.connect("email_analysis.db")
    logs = conn.execute("SELECT timestamp, username, action, resource, result FROM access_logs ORDER BY timestamp DESC").fetchall()
    conn.close()
    return render_template("logs.html", logs=logs, user=user)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        department = request.form['department']
        verified = 'verified' in request.form
        clearance = request.form['clearance']

        password_hash = generate_password_hash(password)
        try:
            conn = sqlite3.connect("email_analysis.db")
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO users (username, password_hash, role, department, verified, clearance)
                              VALUES (?, ?, ?, ?, ?, ?)''',
                           (username, password_hash, role, department, verified, clearance))
            conn.commit()
            flash('Signup successful. Please log in.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists.')
        finally:
            conn.close()

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        conn = sqlite3.connect("email_analysis.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user_row = cursor.fetchone()
        conn.close()

        if user_row and check_password_hash(user_row[2], password):
            session['user'] = {
                'username': user_row[1],
                'role': user_row[3],
                'department': user_row[4],
                'verified': bool(user_row[5]),
                'clearance': user_row[6]
            }
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials.')

    return render_template('login.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username = request.form.get('username')
        conn = sqlite3.connect("email_analysis.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()

        if user:
            temp_password = generate_temp_password()
            hashed = generate_password_hash(temp_password)
            cursor.execute("UPDATE users SET password_hash = ? WHERE username = ?", (hashed, username))
            conn.commit()
            conn.close()
            send_reset_email(username, temp_password)
            flash('Temporary password sent to your email.')
        else:
            flash('Username not found.')
    return render_template('forgot_password.html')

@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('Logged out.')
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
