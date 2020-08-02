from flask import Flask, render_template, request, redirect, session, flash, url_for, send_from_directory
from flask import render_template

app = Flask(__name__)
app.config.from_pyfile('config.py')

from views.mcdm import *

@app.route('/')
def home():
    return render_template('home.html', title='Thor Web')


if __name__ == '__main__':
    app.run(port=5000, debug=True)