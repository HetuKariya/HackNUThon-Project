from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# app.app_context().push()
#
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite3:///database.db'
#
# db = SQLAlchemy(app)
#
# class Item(db.Model):
#     id = db.Column(db.Integer(),primary_key = True)
#     name = db.Column(db.String(length = 30), nullable = False, unique = True)
#     price = db.Column(db.Integer(), nullable = False, unique = True)
#     barcode = db.Column(db.String(length = 20), nullable = False, unique = True)
#     description = db.Column(db.Integer(),nullable = False, unique = True)

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('index.html')
    # return "<h1>Hello World</h1>"

@app.route('/register')
def register_page():
    return render_template("register.html")

@app.route('/login')
def login_page():
    return render_template("login.html")

# dynamic routes
# @app.route('/about/<username>')
# def about_page(username):
#     return f"<h1>This is the about Page of {username}</h1>"

@app.route('/market')
def market_page():
    items = [
        {'id': 1, 'name': 'Phone', 'barcode': '893212299897', 'price': 500},
        {'id': 2, 'name': 'Laptop', 'barcode': '123985473165', 'price': 900},
        {'id': 3, 'name': 'Keyboard', 'barcode': '231985128446', 'price': 150}
    ]
    return render_template("market.html",items = items)

if __name__ == '__main__':
    app.run(debug=True)