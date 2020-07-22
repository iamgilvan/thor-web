from flask import Flask
from flask import render_template

app = Flask(__name__)
app.config.from_pyfile('config.py')

@app.route('/')
def home():
    return ("<p>Hello World</p>")


if __name__ == '__main__':
    app.run(port=5000, debug=True)