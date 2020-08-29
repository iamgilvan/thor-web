from flask import Flask, render_template
import secrets
import os

secret = secrets.token_urlsafe(32)

app = Flask(__name__)
app.config.from_pyfile('config.py')
app.secret_key = os.urandom(24)

from views.mcdm import *

@app.route('/')
def home():
    return render_template('home.html', title='Thor Web')


if __name__ == '__main__':
    app.run(port=5000, debug=True)