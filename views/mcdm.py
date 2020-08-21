from flask import render_template, request, redirect
from app import app
from models.thor import *
from utils.utils import *

thor = Thor()

@app.route('/start')
def start():
    thor.selected_method = int(request.args.get('method'))
    len(thor.alternatives)
    return render_template('main.html', title='Main Parameters',
                            alternative=len(thor.alternatives),
                            criteria=len(thor.criterias),
                            decisor=len(thor.decisors))


@app.route('/main',  methods=['GET', 'POST',])
def main():
    thor.alternatives = [None] * int(request.form['alternative'])
    thor.decisors = [None] * int(request.form['decisor'])
    thor.criterias = [None] * int(request.form['criteria'])
    return render_template('info_names.html', title='Alternatives and Criterias name',
                            alternatives=thor.alternatives,
                            criterias=thor.criterias)


@app.route('/assignment',  methods=['POST'])
def assignment():
    thor.alternatives = [request.form[f'alternative{i}'] for i in range(1, len(thor.alternatives) + 1)]
    thor.criterias = [request.form[f'criteria{i}'] for i in range(1, len(thor.criterias) + 1)]
    # creating the matrices
    thor.matrices = [Utils.create_initial_matrix(len(thor.alternatives)) for i in range(3)]
    thor.weights = [Utils.create_initial_weight(len(thor.criterias)) for i in range(3)]
    return render_template('assignment.html', title='Assignment method')

@app.route('/weight', methods=['POST'])
def weight():
    thor.assignment_method_selected = request.form['assignment']
    # if thor.assignment_method_selected == 'direct':
    # elif thor.assignment_method_selected == 'reason':
    # else: # selected method was interval
    return render_template('weight.html', title='Weight', data=thor)

@app.route('/calculate_weight', methods=['POST'])
def calculate_weight():
    weights_partial = []
    for i in range(1, len(thor.decisors) + 1):
        weights_partial.append([int(request.form[f'value-{i}-{j}']) for j in range(1, len(thor.criterias) + 1)])

    for i in range(len(thor.decisors)):
        norm = max(weights_partial[i])
        for j in range(len(thor.criterias)):
            thor.weights[i][j] += float((weights_partial[i][j]/norm))
    return render_template('p_q_values.html', title='P and Q values', criterias=thor.criterias, weights=thor.weights[0])

@app.route('/disagreement', methods=['POST'])
def disagreement():
    thor.q = [[float(request.form[f'value-q-{i}']) for i in range(1, len(thor.criterias) + 1)]]
    thor.p = [[float(request.form[f'value-p-{i}']) for i in range(1, len(thor.criterias) + 1)]]
    return render_template('disagreement.html', title='Discordancy', criterias=thor.criterias)

@app.route('/pertinence', methods=['POST'])
def pertinence():
    thor.disagreement = [[float(request.form[f'value-d-{i}']) for i in range(1, len(thor.criterias) + 1)]]
    return render_template('pertinence.html', title='Use pertinence?',criterias=thor.criterias)


@app.route('/matrix_alternatives', methods=['POST'])
def matrix_alternatives():
    thor.user_pertinence = True if request.form['pertinence'] == '1' else False
    if thor.user_pertinence:
        thor.pertinence = [[float(request.form[f'value-p-{i}']) for i in range(1, len(thor.criterias) + 1)]]
        thor.pertinence_tca = [[float(request.form[f'value-p-{i}']) for i in range(1, len(thor.criterias) + 1)]]
    else:
        thor.pertinence = [[1 for i in range(1, len(thor.criterias) + 1)]]
        thor.pertinence_tca = [[1 for i in range(1, len(thor.criterias) + 1)]]

    for alt in range(1, len(thor.alternatives) + 1):
        line = [0 for column in range(1 , len(thor.criterias) + 1)]
        thor.main_matrix.append(line)
        thor.pertinence_matrix.append(line)
        thor.pertinence_tca_matrix.append(line)
    return render_template('matrix_alternatives.html', title='Matrix Alternatives', criterias=thor.criterias, alternatives=thor.alternatives)


@app.route('/matrix_pertinence', methods=['POST'])
def matrix_pertinence():
    if not thor.user_pertinence:
        return redirect(url_for('result'))

    index = 1
    for i in range(1, len(thor.alternatives) + 1):
        for j in range(1, len(thor.criterias) + 1):
            thor.main_matrix[i][j]=float((request.form[f'value-{index}-{index}']))
            index+=1
    return render_template('matrix_pertinence.html', title='Matrix Pertinence', criterias=thor.criterias, alternatives=thor.alternatives)

@app.route('/result', methods=['POST'])
def result():
    if thor.user_pertinence:
        index = 1
        for i in range(1, len(thor.alternatives) + 1):
            for j in range(1, len(thor.criterias) + 1):
                thor.pertinence_matrix[i][j]=float((request.form[f'value-{index}-{index}']))
                thor.pertinence_tca_matrix[i][j]=float((request.form[f'value-{index}-{index}']))
                index+=1
    else:
        for i in range(1, len(thor.alternatives) + 1):
            for j in range(1, len(thor.criterias) + 1):
                thor.pertinence_matrix[i][j]=1
                thor.pertinence_tca_matrix[i][j]=1
    return '    '