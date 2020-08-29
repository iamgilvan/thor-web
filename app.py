from flask import Flask, render_template

app = Flask(__name__)
app.config.from_pyfile('config.py')
#app.secret_key = secret

from views.mcdm import *

@app.route('/')
def home():
    return render_template('home.html', title='Thor Web')


@app.route('/start')
def start():
    thor = Thor()
    thor.selected_method = int(request.args.get('method'))
    session['thor'] = json.dumps(thor.__dict__) 
    return render_template('main.html', title='Main Parameters', alternative=len(thor.alternatives), criteria=len(thor.criterias), decisor=len(thor.decisors))


@app.route('/main',  methods=['GET', 'POST',])
def main():
    thor = Payload(session.get('thor', None))
    thor.alternatives = [None] * int(request.form['alternative'])
    thor.decisors = [None] * int(request.form['decisor'])
    thor.criterias = [None] * int(request.form['criteria'])
    session['thor'] = json.dumps(thor.__dict__)
    return render_template('info_names.html', title='Alternatives and Criterias name',
                            alternatives=thor.alternatives,
                            criterias=thor.criterias)


@app.route('/weight', methods=['POST'])
def weight():
    thor = Payload(session.get('thor', None))
    thor.alternatives = [request.form[f'alternative{i}'] for i in range(1, len(thor.alternatives) + 1)]
    thor.criterias = [request.form[f'criteria{i}'] for i in range(1, len(thor.criterias) + 1)]
    session['thor'] = json.dumps(thor.__dict__)
    return render_template("weight.html", title='Weight', criterias=thor.criterias, alternatives=thor.alternatives, decisors=thor.decisors)


if __name__ == '__main__':
    app.run(port=5000, debug=True)