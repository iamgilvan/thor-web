from flask import Flask, render_template

app = Flask(__name__)
app.config.from_pyfile('config.py')
#app.secret_key = secret

from views.mcdm import *

@app.route('/')
def home():
    return render_template('home.html', title='Thor Web')


if __name__ == '__main__':
    app.run(port=5000, debug=True)