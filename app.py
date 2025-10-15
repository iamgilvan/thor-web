from flask import Flask, render_template, redirect, url_for, request
from utils import mongo_utils as mu
from models.video import *
import datetime
global config
app = Flask(__name__)
app.config.from_pyfile('config.py')

from views.mcdm import *

configuration_file = "./input/config.json"

import json


def build_config_params(full_path):
    global config
    with open(full_path) as f:
        config = json.loads(f.read())

def get_collection(col):
    collection = None
    while collection is None:
        build_config_params(configuration_file)
        collection = mu.connection(col)
    return collection


@app.route('/')
def home():
    return render_template('home.html', title='Thor Web')

@app.route('/maintenance')
def maintenance():
    return render_template('maintenance.html', title='Articles')

# Route for handling the login page logic
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        collection = get_collection('users')
        user = collection.find_one({'username': request.form['username']})
        if request.form['username'] != user['username'] or request.form['password'] != user['password']:
            error = 'Invalid Credentials. Please try again.'
        else:
            return render_template('register.html', title='Register video')
    return render_template('login.html', error=error)

@app.route('/delete', methods=['POST'])
def delete():
    try:
        collection = get_collection('videos')
        url = request.form['url']
        collection.delete_one({'url': url})
        message = 'Deleted Successfully'
    except:
        message = "Can't find video"
    return render_template('register.html', title='Register video', message=message)

@app.route('/register', methods=['POST'])
def register():
    collection = get_collection('videos')
    try:
        video = Video()
        video.title = request.form['title']
        video.url = request.form['url']
        video.description = request.form['description']
        video.publishdate = datetime.strptime(request.form['publishdate'], '%Y-%m-%d')
        video.publishdateformat = request.form['publishdate']
        if 'embed' not in video.url:
            return render_template('register.html', title='Register video', message='Incorrect url')
        collection.insert_one(video.__dict__)
        message = 'Inserted Successfully'
    except:
        message = 'Erro to insert video'
    return render_template('register.html', title='Register video', message=message)

@app.route('/usecases')
def usecase():
    collection = get_collection('videos')
    cases = collection.find().sort("publishdate", -1)
    return render_template('usecases.html', title='Use Cases', cases=cases)


@app.route('/about')
def about():
    return render_template('about.html', title='About')


@app.route('/registerfile')
def registerfile():
    return render_template('register-page.html', title='Software Registrations')


if __name__ == '__main__':
    app.run(port=5000)