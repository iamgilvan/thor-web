from flask import render_template, request, redirect, session, url_for, flash, jsonify
from app import app
from models.thor import *
from utils.utils import *
import json
from bson.objectid import ObjectId
from utils import mongo_utils as mu
from utils.pdf_utils import PDF
import os
from datetime import datetime
import numpy as np

configuration_file = "./input/config.json"

import json
global config

def build_config_params(full_path):
    global config
    with open(full_path) as f:
        config = json.loads(f.read())

@app.route('/start')
def start():
    collection = None
    while collection is None:
        build_config_params(configuration_file)
        collection = mu.open_mongo_connection(config['mongo']['thor'])
    thor = Thor()
    thor.selected_method = int(request.args.get('method'))
    id = mu.write_on_mongo(collection, thor)
    for file in os.listdir('static'):
        if file.endswith('.pdf'):
            os.remove('static/' + file)
    return render_template('main.html',
                            title='Main Parameters',
                            id=id)

@app.route('/main/<string:id>',  methods=['POST',])
def main(id):
    collection = None
    while collection is None:
        build_config_params(configuration_file)
        collection = mu.open_mongo_connection(config['mongo']['thor'])
    thor = mu.get_objects(collection, ObjectId(id))
    thor['alternatives'] = [None] * int(request.form['alternative'])
    thor['decisors'] = [None] * int(request.form['decisor'])
    thor['criterias'] = [None] * int(request.form['criteria'])

    mu.update(ObjectId(id), thor, collection)
    return render_template('info_names.html',
                           id=id,
                            title='Alternatives and Criterias name',
                            alternatives=thor['alternatives'],
                            criterias=thor['criterias'])

@app.route('/weight/<string:id>', methods=['POST'])
def weight(id):
    collection = None
    while collection is None:
        build_config_params(configuration_file)
        collection = mu.open_mongo_connection(config['mongo']['thor'])
    
    thor = mu.get_objects(collection, ObjectId(id))
    thor['assignment_method_selected'] = 0
    thor['alternatives'] = [request.form[f'alternative{i}'] for i in range(1, len(thor['alternatives']) + 1)]
    thor['criterias'] = [request.form[f'criteria{i}'] for i in range(1, len(thor['criterias']) + 1)]

    cri = len(thor['criterias'])
    questions = []
    for j in range(cri-1):
        questions.append(thor['criterias'][j+1] + ' regarding ' + thor['criterias'][j])
    thor['questions'] = questions

    mu.update(ObjectId(id), thor, collection)


    if thor['selected_method'] == 3:
        return render_template("fuzzy_weight_choice.html", id=id,
                            title='Weight')
    
    return render_template("weight.html",
                            questions=questions,
                            id=id,
                            title='Weight',
                            criterias=thor['criterias'],
                            alternatives=thor['alternatives'],
                            decisors=thor['decisors'])

def build_ahp_matrix(criterias, form, d_index):
    """
    Constrói a matriz AHP NxN para um decisor (d_index) a partir dos campos do form:
    preference_{i+1}_{j+1}_{d_index}, somente para j>i.
    Diagonal = 1.0, parte inferior = recíprocos.
    """
    import math

    n = len(criterias)
    allowed_values = set(range(1, 10))  # 1..9

    # Inicializa NxN
    AHP_matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if j > i:
                field = f"preference_{i+1}_{j+1}_{d_index}"
                if field not in form or form[field] == "":
                    raise ValueError(f"Preencha todos os valores. Campo faltando: {field}")
                try:
                    v = int(form[field])
                except ValueError:
                    raise ValueError(f"Valor inválido em {field}. Use números inteiros de 1 a 9.")

                if v not in allowed_values:
                    raise ValueError(f"Valor fora do permitido em {field}. Use inteiros de 1 a 9.")

                AHP_matrix[i][j] = float(v)

            elif j < i:
                # Recíproco do elemento simétrico (preenchido quando j>i)
                if AHP_matrix[j][i] == 0.0:
                    # Ainda não definido porque o item simétrico fica no triângulo superior — ele será definido
                    # quando (j,i) for visitado; então só pula aqui e define depois.
                    pass
                else:
                    AHP_matrix[i][j] = 1.0 / AHP_matrix[j][i]
            else:
                # Diagonal principal
                AHP_matrix[i][j] = 1.0

    # Garantir que todos os recíprocos foram preenchidos
    for i in range(n):
        for j in range(n):
            if j < i and AHP_matrix[i][j] == 0.0:
                if AHP_matrix[j][i] == 0.0:
                    raise ValueError("Matriz incompleta: entradas do triângulo superior faltando.")
                AHP_matrix[i][j] = 1.0 / AHP_matrix[j][i]

    return AHP_matrix
triangular_membership_function = {1: [1, 1, 1], 2: [1, 2, 3], 3: [2, 3, 4], 4: [3, 4, 5], 5: [4, 5, 6],
                                  6: [5, 6, 7], 7: [6, 7, 8],
                                  8: [7, 8, 9], 9: [9, 9, 9]}
trapezoidal_membership_function = {1: [1, 1, 1, 1], 2: [1, 1.5, 2.5, 3], 3: [2, 2.5, 3.5, 4], 4: [3, 3.5, 4.5, 5],
                                   5: [4, 4.5, 5.5, 6],
                                   6: [5, 5.5, 6.5, 7], 7: [6, 6.5, 7.5, 8], 8: [7, 7.5, 8.5, 9], 9: [8, 8.5, 9, 9]}

def consistency(AHP_matrix):
    n = len(AHP_matrix)
    triangular_fuzzy_matrix = np.zeros((n, n, 3))
    trapezoidal_fuzzy_matrix = np.zeros((n, n, 4))

    # Defining triangular consistency
    for x in range(n):  # Creating the triangular fuzzy matrix
        for y in range(n):
            if AHP_matrix[x][y] >= 1:
                triangular_fuzzy_matrix[x][y] = triangular_membership_function[round(AHP_matrix[x][y])]
            else:
                index = round(1 / AHP_matrix[x][y])
                temp = triangular_membership_function[index]
                for i in range(3):
                    triangular_fuzzy_matrix[x][y][i] = 1.0 / temp[2 - i]

    # Defining the terms
    dimension = triangular_fuzzy_matrix.shape[0]
    average_matrix = np.zeros((dimension, dimension))
    average_sum = np.zeros(dimension)

    # Calculating the average matrix triangular
    for i in range(dimension):
        for j in range(dimension):
            average_matrix[i][j] = np.mean(triangular_fuzzy_matrix[i][j])

    # Calculating the sum of the columns of the average matrix
    for i in range(dimension):
        sum = 0
        for j in range(dimension):
            sum += average_matrix[j][i]
        average_sum[i] = sum

    # Calculating the normalized matrix
    normalized_matrix = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            normalized_matrix[i][j] = average_matrix[i][j] / average_sum[j]

    # Calculating the average vector normalized
    average_vector_normalized = np.zeros(dimension)
    for i in range(dimension):
        average_vector_normalized[i] = np.mean(normalized_matrix[i])

    # Calculating the final matrix
    final_matrix = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            final_matrix[i][j] = average_vector_normalized[j] * average_matrix[i][j]

    # Calculating the final sum vector and the final vector
    final_sum_vector = np.sum(final_matrix, axis=1)
    final_vector = np.zeros(dimension)
    for i in range(dimension):
        final_vector[i] = final_sum_vector[i] / average_vector_normalized[i]

    # Calculating the final terms
    final_sum = np.sum(final_vector, axis=0)
    lamb = final_sum / dimension
    table_ri = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
    ri = table_ri[dimension - 1]
    ci = (lamb - dimension) / (dimension - 1)
    cr = ci / ri

    # Defining trapezoidal consistency
    for x in range(n):  # Creating the trapezoidal fuzzy matrices
        for y in range(n):
            if AHP_matrix[x][y] >= 1:
                trapezoidal_fuzzy_matrix[x][y] = trapezoidal_membership_function[round(AHP_matrix[x][y])]
            else:
                index = round(1 / AHP_matrix[x][y])
                temp = trapezoidal_membership_function[index]
                for i in range(4):
                    trapezoidal_fuzzy_matrix[x][y][i] = 1.0 / temp[3 - i]

    # Defining the terms
    dimension = trapezoidal_fuzzy_matrix.shape[0]
    average_matrix_trapezoidal = np.zeros((dimension, dimension))
    average_sum_trapezoidal = np.zeros(dimension)

    # Calculating the average matrix
    for i in range(dimension):
        for j in range(dimension):
            average_matrix_trapezoidal[j][i] = (trapezoidal_fuzzy_matrix[j][i][0] + (
                    2 * trapezoidal_fuzzy_matrix[j][i][1]) + (2 * trapezoidal_fuzzy_matrix[j][i][2]) +
                                                trapezoidal_fuzzy_matrix[j][i][3]) / 6

    # Calculating the sum of the columns of the average matrix
    for i in range(dimension):
        sum_trapezoidal = 0
        for j in range(dimension):
            sum_trapezoidal += average_matrix_trapezoidal[j][i]
        average_sum_trapezoidal[i] = sum_trapezoidal

    # Calculating the normalized matrix
    normalized_matrix_trapezoidal = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            normalized_matrix_trapezoidal[i][j] = average_matrix_trapezoidal[i][j] / average_sum_trapezoidal[j]

    # Calculating the average vector normalized
    average_vector_normalized_trapezoidal = np.zeros(dimension)
    for i in range(dimension):
        average_vector_normalized_trapezoidal[i] = np.mean(normalized_matrix_trapezoidal[i])

    # Calculating the final matrix
    final_matrix_trapezoidal = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            final_matrix_trapezoidal[i][j] = average_vector_normalized_trapezoidal[j] * \
                                             average_matrix_trapezoidal[i][j]

    # Calculating the final sum vector and the final vector
    final_sum_vector_trapezoidal = np.sum(final_matrix_trapezoidal, axis=1)
    final_vector_trapezoidal = np.zeros(dimension)
    for i in range(dimension):
        final_vector_trapezoidal[i] = final_sum_vector_trapezoidal[i] / average_vector_normalized_trapezoidal[i]

    # Calculating the final terms
    final_sum_trapezoidal = np.sum(final_vector_trapezoidal, axis=0)
    lamb_trapezoidal = final_sum_trapezoidal / dimension
    ci_trapezoidal = (lamb_trapezoidal - dimension) / (dimension - 1)
    cr_trapezoidal = ci_trapezoidal / ri

    return cr, cr_trapezoidal
# Creating the fuzzy matrices

def fuzzy_AHP(all_AHP_matrices):
    all_fuzzified_matrices_triangular = []
    all_fuzzified_matrices_trapezoidal = []

    for idx, AHP_matrix in enumerate(all_AHP_matrices, start=1):
        test_data = AHP_matrix
        n = len(test_data)
        fuzzified_test_data_triangular = np.zeros((n, n, 3))
        fuzzified_test_data_trapezoidal = np.zeros((n, n, 4))

        # Creating the triangular fuzzy matrices
        for x in range(n):
            for y in range(n):
                if test_data[x][y] >= 1:
                    fuzzified_test_data_triangular[x][y] = triangular_membership_function[round(test_data[x][y])]
                else:
                    index = round(1 / test_data[x][y])
                    temp = triangular_membership_function[index]
                    for i in range(3):
                        fuzzified_test_data_triangular[x][y][i] = 1.0 / temp[2 - i]
        all_fuzzified_matrices_triangular.append(fuzzified_test_data_triangular)

        # Creating the trapezoidal fuzzy matrices
        for x in range(n):
            for y in range(n):
                if test_data[x][y] >= 1:
                    fuzzified_test_data_trapezoidal[x][y] = trapezoidal_membership_function[round(test_data[x][y])]
                else:
                    index = round(1 / test_data[x][y])
                    temp = trapezoidal_membership_function[index]
                    for i in range(4):
                        fuzzified_test_data_trapezoidal[x][y][i] = 1.0 / temp[3 - i]
        all_fuzzified_matrices_trapezoidal.append(fuzzified_test_data_trapezoidal)

    return all_fuzzified_matrices_triangular, all_fuzzified_matrices_trapezoidal  # Returns all triangular and trapezoidal fuzzy matrices
# Creating a new fuzzy triangular matrix

def create_new_triangular(all_fuzzified_matrices_triangular):  # Creating the final triangular matrix
    num_matrices = len(all_fuzzified_matrices_triangular)
    n = len(all_fuzzified_matrices_triangular[0])  # Assuming all matrices have the same size
    new_matrix_fuzzy_triangular = np.zeros((n, n, 3))  # Initialize the new matrix with zeros

    # Taking geometric means to obtain the terms of the new triangular fuzzy matrix
    for x in range(n):
        for y in range(n):
            product = 1.0
            for matrix in all_fuzzified_matrices_triangular:
                product *= matrix[x][y][0]  # Multiplying the first element of each fuzzified matrix
            new_matrix_fuzzy_triangular[x][y][0] = product ** (1.0 / num_matrices)  # Taking the geometric mean
            for i in range(1,
                           3):  # Now calculate the geometric mean for the other two elements (second and third) of the fuzzy values
                geometric_mean = 1.0
                for matrix in all_fuzzified_matrices_triangular:
                    geometric_mean *= matrix[x][y][i]
                new_matrix_fuzzy_triangular[x][y][i] = geometric_mean ** (1.0 / num_matrices)

    return new_matrix_fuzzy_triangular  # Returns the final triangular fuzzy matrix
# Creating a new fuzzy trapezoidal matrix

def create_new_trapezoidal(all_fuzzified_matrices_trapezoidal):
    num_matrices = len(all_fuzzified_matrices_trapezoidal)
    n = len(all_fuzzified_matrices_trapezoidal[0])  # Assuming all matrices have the same size
    new_matrix_fuzzy_trapezoidal = np.zeros((n, n, 4))  # Initialize the new matrix with zeros

    # Taking geometric means to obtain the terms of the new triangular fuzzy matrix
    for x in range(n):
        for y in range(n):
            product = 1.0
            for matrix in all_fuzzified_matrices_trapezoidal:
                product *= matrix[x][y][0]  # Multiplying the first element of each fuzzified matrix
            new_matrix_fuzzy_trapezoidal[x][y][0] = product ** (1.0 / num_matrices)  # Taking the geometric mean

            for i in range(1,
                           4):  # Now calculate the geometric mean for the other three elements (second, third, and fourth) of the fuzzy values
                geometric_mean = 1.0
                for matrix in all_fuzzified_matrices_trapezoidal:
                    geometric_mean *= matrix[x][y][i]
                new_matrix_fuzzy_trapezoidal[x][y][i] = geometric_mean ** (1.0 / num_matrices)

    return new_matrix_fuzzy_trapezoidal  # Returns the final trapezoidal fuzzy matrix
# Defining the normalized weights

def fuzzy_geometric_mean(new_matrix_fuzzy_triangular, new_matrix_fuzzy_trapezoidal):
    n = len(new_matrix_fuzzy_triangular)
    fuzzy_geometric_mean_triangular = [[1 for x in range(3)] for y in
                                       range(n)]  # Creating Fuzzy Geometric Mean Triangular
    fuzzy_geometric_mean_trapezoidal = [[1 for x in range(4)] for y in
                                        range(n)]  # Creating Fuzzy Geometric Mean Trapezoidal

    # Obtaining the values of the Fuzzy Geometric Mean Triangular
    for i in range(n):  # Taking the geometric mean
        for j in range(3):
            for k in range(n):
                fuzzy_geometric_mean_triangular[i][j] *= new_matrix_fuzzy_triangular[i][k][j]
            fuzzy_geometric_mean_triangular[i][j] = fuzzy_geometric_mean_triangular[i][j] ** (1 / float(n))

    # Obtaining the values of the Fuzzy Geometric Mean Trapezoidal
    for i in range(n):  # Taking the geometric mean
        for j in range(4):
            for k in range(n):
                fuzzy_geometric_mean_trapezoidal[i][j] *= new_matrix_fuzzy_trapezoidal[i][k][j]
            fuzzy_geometric_mean_trapezoidal[i][j] = fuzzy_geometric_mean_trapezoidal[i][j] ** (1 / float(n))

    sum_fuzzy_gm_triangular = [0 for x in range(
        3)]  # Creating the empty vector sum of the fuzzy geometric mean value triangular
    inv_sum_fuzzy_gm_triangular = [0 for x in range(
        3)]  # Creating an empty vector with the inverse of the sum of the fuzzy geometric mean value triangular

    # Adding the columns of the fuzzy geometric mean value triangular
    for i in range(3):
        for j in range(n):
            sum_fuzzy_gm_triangular[i] += fuzzy_geometric_mean_triangular[j][i]
    # Obtaining the inverse of the sum of the columns of the fuzzy geometric mean value triangular
    for i in range(3):
        inv_sum_fuzzy_gm_triangular[i] = (1.0 / sum_fuzzy_gm_triangular[2 - i])

    fuzzy_weights_triangular = [[1 for x in range(3)] for y in range(n)]  # Creating the fuzzy weights matrix

    # Filling the fuzzy weights matrix
    for i in range(n):
        for j in range(3):
            fuzzy_weights_triangular[i][j] = fuzzy_geometric_mean_triangular[i][j] * inv_sum_fuzzy_gm_triangular[j]

    weights_triangular = [0 for i in range(n)]  # Creating the empty vector of weights
    normalized_weights_triangular = [0 for i in range(n)]  # Creating the empty vector from the normalized weights
    sum_weights_triangular = 0  # Initial value of the sum of weights

    # Filling the vector of weights
    for i in range(n):
        for j in range(3):
            weights_triangular[i] += fuzzy_weights_triangular[i][j]
        weights_triangular[i] /= 3
        sum_weights_triangular += weights_triangular[i]

    # Filling the vector of normalized weights
    for i in range(n):
        normalized_weights_triangular[i] = (1.0 * weights_triangular[i]) / (1.0 * sum_weights_triangular)

    sum_fuzzy_gm_trapezoidal = [0 for x in range(
        4)]  # Creating the empty vector sum of the fuzzy geometric mean value trapezoidal
    inv_sum_fuzzy_gm_trapezoidal = [0 for x in range(
        4)]  # Creating an empty vector with the inverse of the sum of the fuzzy geometric mean value trapezoidal

    # Adding the columns of the fuzzy geometric mean value trapezoidal
    for i in range(4):
        for j in range(n):
            sum_fuzzy_gm_trapezoidal[i] += fuzzy_geometric_mean_trapezoidal[j][i]
    # Obtaining the inverse of the sum of the columns of the fuzzy geometric mean value
    for i in range(4):
        inv_sum_fuzzy_gm_trapezoidal[i] = (1.0 / sum_fuzzy_gm_trapezoidal[3 - i])

    fuzzy_weights_trapezoidal = [[1 for x in range(4)] for y in range(n)]  # Creating the fuzzy weights matrix

    # Filling the fuzzy weights matrix
    for i in range(n):
        for j in range(4):
            fuzzy_weights_trapezoidal[i][j] = fuzzy_geometric_mean_trapezoidal[i][j] * inv_sum_fuzzy_gm_trapezoidal[
                j]

    weights_trapezoidal = [0 for i in range(n)]  # Creating the empty vector of weights
    normalized_weights_trapezoidal = [0 for i in range(n)]  # Creating the empty vector from the normalized weights
    sum_weights_trapezoidal = 0  # Initial value of the sum of weights

    # Filling the vector of weights
    for i in range(n):
        for j in range(4):
            if j == 1 or j == 2:
                weights_trapezoidal[i] += 2 * fuzzy_weights_trapezoidal[i][j]
            else:
                weights_trapezoidal[i] += fuzzy_weights_trapezoidal[i][j]
        weights_trapezoidal[i] /= 6  # 2 (second column) + 2 (third column) + 1 (first and fourth column)
        sum_weights_trapezoidal += weights_trapezoidal[i]

    # Filling the vector of normalized weights
    for i in range(n):
        normalized_weights_trapezoidal[i] = (1.0 * weights_trapezoidal[i]) / (1.0 * sum_weights_trapezoidal)

    return normalized_weights_triangular, normalized_weights_trapezoidal, fuzzy_geometric_mean_triangular, fuzzy_geometric_mean_trapezoidal,fuzzy_weights_triangular, fuzzy_weights_trapezoidal,weights_triangular, weights_trapezoidal  # Returns the triangular and trapezoidal normalized weight vector

@app.route('/fuzzy_ahp/<string:id>', methods=['GET'])
def fuzzy_ahp(id):
    collection = None
    while collection is None:
        build_config_params(configuration_file)
        collection = mu.open_mongo_connection(config['mongo']['thor'])

    thor = mu.get_objects(collection, ObjectId(id))
    criterias = thor.get('criterias', [])
    decisors = thor.get('decisors', [])
    thor['assignment_method_selected'] = "Fuzzy AHP"

    d_index = int(request.args.get('dm', 1)) 
    if d_index > len(decisors):

        return redirect(url_for('fuzzy_ahp_results', id=id))

    current_decisor = decisors[d_index - 1]
    mu.update(ObjectId(id), thor, collection)
    return render_template(
        "fuzzy_ahp.html",
        id=id,
        title='Fuzzy AHP',
        criterias=criterias,
        decisors=[current_decisor],  
        d_index=d_index,
        total_decisors=len(decisors),
        
    )

@app.route('/fuzzy_ahp_results/<string:id>', methods=['GET'])
def fuzzy_ahp_results(id):
    collection = None
    while collection is None:
        build_config_params(configuration_file)
        collection = mu.open_mongo_connection(config['mongo']['thor'])

    thor = mu.get_objects(collection, ObjectId(id))
    mu.update(ObjectId(id), thor, collection)
    preferences = thor.get('preferences', [])
    ahp_matrices = thor.get('ahp_matrices', {})
    criterias = thor.get('criterias', [])

    triangular_weights = list(zip(criterias, thor.get('fuzzy_weights', {}).get('triangular', [])))
    trapezoidal_weights = list(zip(criterias, thor.get('fuzzy_weights', {}).get('trapezoidal', [])))
    new_matrix_fuzzy_triangular = thor.get('new_matrix_fuzzy_triangular')
    new_matrix_fuzzy_trapezoidal = thor.get('new_matrix_fuzzy_trapezoidal')
    cr = thor.get('cr')
    cr_trapezoidal = thor.get('cr_trapezoidal')
    fuzzy_geometric_mean_triangular = thor.get('fuzzy_geometric_mean_triangular')
    fuzzy_geometric_mean_trapezoidal = thor.get('fuzzy_geometric_mean_trapezoidal')
    fuzzy_weights_triangular = thor.get('fuzzy_weights_triangular')
    fuzzy_weights_trapezoidal = thor.get('fuzzy_weights_trapezoidal')
    weights_triangular = thor.get('weights_triangular')
    weights_trapezoidal = thor.get('weights_trapezoidal')
    return render_template(
        'fuzzy_ahp_results.html',
        id=id,
        title='Results Fuzzy AHP',
        preferences=preferences,
        ahp_matrices=ahp_matrices,
        criterias=criterias,
        thor=thor,
        triangular_weights=triangular_weights,
        trapezoidal_weights=trapezoidal_weights,
        new_matrix_fuzzy_triangular=new_matrix_fuzzy_triangular,
        new_matrix_fuzzy_trapezoidal=new_matrix_fuzzy_trapezoidal,
        cr=cr,
        cr_trapezoidal=cr_trapezoidal,
        fuzzy_geometric_mean_triangular=fuzzy_geometric_mean_triangular,
        fuzzy_geometric_mean_trapezoidal=fuzzy_geometric_mean_trapezoidal,
        fuzzy_weights_triangular=fuzzy_weights_triangular,
        fuzzy_weights_trapezoidal=fuzzy_weights_trapezoidal,
        weights_triangular=weights_triangular,
        weights_trapezoidal=weights_trapezoidal
    )

DATA_AHP = [
    ("Extremely Weak", "0.111"),
    ("Very Weak / Extremely Weak", "0.125"),
    ("Very Weak", "0.143"),
    ("Weak / Very Weak", "0.167"),
    ("Weak", "0.200"),
    ("Moderate Weak / Weak", "0.250"),
    ("Moderate Weak", "0.333"),
    ("Equal / Moderate Weak", "0.500"),
    ("Equal", "1.00"),
    ("Equal / Moderate Strong", "2.00"),
    ("Moderate Strong", "3.00"),
    ("Moderate Strong / Strong", "4.00"),
    ("Strong", "5.00"),
    ("Strong / Very Strong", "6.00"),
    ("Very Strong", "7.00"),
    ("Very Strong / Extremely Strong", "8.00"),
    ("Extremely Strong", "9.00"),
]
ALLOWED_STRINGS = {v for _, v in DATA_AHP}

@app.route('/save_fuzzy_ahp/<string:id>', methods=['POST'])
def save_fuzzy_ahp(id):
    build_config_params(configuration_file) 
    collection = mu.open_mongo_connection(config['mongo']['thor'])
    
    thor = mu.get_objects(collection, ObjectId(id))
    criterias = thor.get('criterias', [])
    decisors = thor.get('decisors', [])

    d_index = int(request.form.get('d_index', 1))
    current_decisor = decisors[d_index - 1] or f"Decision Maker {d_index}"

    n = len(criterias)
    AHP_matrix = np.ones((n, n), dtype=float)
    preferences_to_save = []

    for i in range(n):
        for j in range(n):
            if j > i:
                field_name = f"preference_{i+1}_{j+1}_{d_index}"
                raw = request.form.get(field_name, "").strip()
                if raw not in ALLOWED_STRINGS:
                    flash(
                        f"Decision maker {d_index}: valor inválido '{raw}' em {criterias[i]} vs {criterias[j]}.",
                        "danger",
                    )
                    return redirect(url_for('fuzzy_ahp', id=id, dm=d_index))
                value = float(raw)  
                AHP_matrix[i, j] = value
                AHP_matrix[j, i] = 1.0 / value
                preferences_to_save.append({
                    "decisor": f"Decision Maker {d_index}",
                    "criteria_i": criterias[i],
                    "criteria_j": criterias[j],
                    "value": value
                })

    cr, cr_trapezoidal = consistency(AHP_matrix)

    if cr > 0.1 or cr_trapezoidal > 0.1:
        flash(
            f"Decision maker {d_index}: A matrix is not consistent "
            f"(Triangular consistency:={cr:.3f}, Trapezoidal consistency:{cr_trapezoidal:.3f}). "
            "Please review your judgments.",
            "danger",
        )
        return redirect(url_for('fuzzy_ahp', id=id, dm=d_index))

    if 'preferences' not in thor:
        thor['preferences'] = []
    thor['preferences'].extend(preferences_to_save)

    if 'ahp_matrices' not in thor:
        thor['ahp_matrices'] = {}
    thor['ahp_matrices'][current_decisor] = AHP_matrix.tolist()

    mu.update(ObjectId(id), thor, collection)

    if d_index < len(decisors):
        return redirect(url_for('fuzzy_ahp', id=id, dm=d_index+1))
    else:
        all_AHP_matrices = [np.array(m) for m in thor.get('ahp_matrices', {}).values()]
        all_fuzzified_matrices_triangular, all_fuzzified_matrices_trapezoidal = fuzzy_AHP(all_AHP_matrices)
        new_matrix_fuzzy_triangular = create_new_triangular(all_fuzzified_matrices_triangular)
        new_matrix_fuzzy_trapezoidal = create_new_trapezoidal(all_fuzzified_matrices_trapezoidal)
        result_triangular, result_trapezoidal, fuzzy_geometric_mean_triangular, fuzzy_geometric_mean_trapezoidal,fuzzy_weights_triangular, fuzzy_weights_trapezoidal, weights_triangular, weights_trapezoidal = fuzzy_geometric_mean(new_matrix_fuzzy_triangular, new_matrix_fuzzy_trapezoidal)

        thor['fuzzy_weights'] = {
            'triangular': [round(w, 3) for w in result_triangular],
            'trapezoidal': [round(w, 3) for w in result_trapezoidal]
        }
        thor['new_matrix_fuzzy_triangular'] = np.round(new_matrix_fuzzy_triangular, 3).tolist()
        thor['new_matrix_fuzzy_trapezoidal'] = np.round(new_matrix_fuzzy_trapezoidal, 3).tolist()
        thor['cr'] = round(cr, 3)
        thor['cr_trapezoidal'] = round(cr_trapezoidal, 3)
        thor['fuzzy_geometric_mean_triangular'] = np.round(fuzzy_geometric_mean_triangular, 3).tolist()
        thor['fuzzy_geometric_mean_trapezoidal'] = np.round(fuzzy_geometric_mean_trapezoidal, 3).tolist()
        thor['fuzzy_weights_triangular'] = np.round(fuzzy_weights_triangular, 3).tolist()
        thor['fuzzy_weights_trapezoidal'] = np.round(fuzzy_weights_trapezoidal, 3).tolist()
        thor['weights_triangular'] = np.round(weights_triangular, 3).tolist()
        thor['weights_trapezoidal'] = np.round(weights_trapezoidal, 3).tolist()
        mu.update(ObjectId(id), thor, collection)

        return redirect(url_for('fuzzy_ahp_results',
                                id=id
                                )
        )

@app.route('/api/fuzzy_ahp/check/<string:id>', methods=['POST'])
def api_check_fuzzy_ahp(id):

    try:
        build_config_params(configuration_file)
        collection = mu.open_mongo_connection(config['mongo']['thor'])

        thor = mu.get_objects(collection, ObjectId(id))
        criterias = thor.get('criterias', [])
        if not criterias:
            return jsonify(ok=False, message="Critérios não encontrados."), 200

        n = len(criterias)
        try:
            d_index = int(request.form.get('d_index', 1))
        except Exception:
            d_index = 1

        
        AHP_matrix = np.ones((n, n), dtype=float)
        missing = []

        for i in range(n):
            for j in range(n):
                if j > i:
                    field_name = f"preference_{i+1}_{j+1}_{d_index}"
                    raw = (request.form.get(field_name, "") or "").strip()
                    if not raw:
                        missing.append(field_name)
                        continue
                    if raw not in ALLOWED_STRINGS:
                        
                        return jsonify(
                            ok=False,
                            message=f"Valor inválido '{raw}' em {criterias[i]} vs {criterias[j]}.",
                            field=field_name
                        ), 200
                    val = float(raw)
                    AHP_matrix[i, j] = val
                    AHP_matrix[j, i] = 1.0 / val

        if missing:
            return jsonify(ok=False, message="Campos faltando.", missing=missing, allFilled=False), 200

        
        cr, cr_trapezoidal = consistency(AHP_matrix)
        consistent = (cr <= 0.1 and cr_trapezoidal <= 0.1)

        return jsonify(
            ok=True,
            cr=round(float(cr), 6),
            cr_trapezoidal=round(float(cr_trapezoidal), 6),
            consistent=bool(consistent)
        ), 200

    except Exception as e:
        
        return jsonify(ok=False, message=str(e)), 200

@app.route('/fuzzy_ahp_results_submit/<string:id>', methods=['POST'])
def fuzzy_ahp_results_submit(id):

    build_config_params(configuration_file)
    collection = mu.open_mongo_connection(config['mongo']['thor'])
    thor = mu.get_objects(collection, ObjectId(id))

    if 'fuzzy_weights' in thor:

        thor['peso_triangular'] = thor['fuzzy_weights']['triangular']
        thor['peso_trapezoidal'] = thor['fuzzy_weights']['trapezoidal']

     
        num_decisors = len(thor['decisors'])
        thor['pesofim'] = [thor['peso_triangular'] for _ in range(num_decisors)]
        thor['peso'] = thor['peso_triangular']

        mu.update(ObjectId(id), thor, collection)

        return render_template('matrix.html',
                            id=id,
                            title='Matrix',
                            peso=thor['peso'],
                            pesotrap = thor['peso_trapezoidal'],
                            pesofims=thor['pesofim'],
                            criterias=thor['criterias'],
                            alternatives=thor['alternatives'],
                            decisors=thor['decisors'],
                            thor=thor)


TRI_DICT = {0: [0.0, 0.0, 0.25], 1: [0.0, 0.25, 0.5], 2: [0.25, 0.5, 0.75], 3: [0.5, 0.75, 1.0], 4: [0.75, 1.0, 1.0]}
TRAP_DICT = {0: [0.0, 0.0, 0.1, 0.2], 1: [0.1, 0.2, 0.3, 0.4], 2: [0.3, 0.4, 0.5, 0.6], 3: [0.5, 0.6, 0.7, 0.8], 4: [0.7, 0.8, 0.9, 1.0]}
DEMATEL_ALLOWED = {"0","1","2","3","4"}

def _defuzz_tri_mean(v3):  
    return (float(v3[0]) + float(v3[1]) + float(v3[2])) / 3.0

def _defuzz_trap_61216(v4):  
    return (float(v4[0]) + 2*float(v4[1]) + 2*float(v4[2]) + float(v4[3])) / 6.0

def _normalize_by_biggest_sum_list(M):
    row_sums = [sum(r) for r in M]
    col_sums = [sum(c) for c in zip(*M)]
    biggest = max(max(row_sums), max(col_sums)) if row_sums and col_sums else 1.0
    return [[v / biggest for v in r] for r in M] if biggest != 0 else [r[:] for r in M]

def _compute_dematel_weights_exact(tri_mats, trap_mats):
    import numpy as np
    tri_mats = [np.array(m, dtype=float) for m in tri_mats]
    trap_mats = [np.array(m, dtype=float) for m in trap_mats]
    n = tri_mats[0].shape[0]


    tri_agg = np.mean(tri_mats, axis=0)   
    trap_agg = np.mean(trap_mats, axis=0) 


    tri_def = [[_defuzz_tri_mean(tri_agg[i, j]) for j in range(n)] for i in range(n)]
    trap_def = [[_defuzz_trap_61216(trap_agg[i, j]) for j in range(n)] for i in range(n)]


    N_tri  = _normalize_by_biggest_sum_list(tri_def)
    N_trap = _normalize_by_biggest_sum_list(trap_def)

    N_tri_np  = np.array(N_tri, dtype=float)
    N_trap_np = np.array(N_trap, dtype=float)

    I_tri = np.eye(n); I_trap = np.eye(n)
    A_tri = I_tri - N_tri_np; A_trap = I_trap - N_trap_np
  
    try:
        inv_tri = np.linalg.inv(A_tri)
    except np.linalg.LinAlgError:
        inv_tri = np.linalg.pinv(A_tri)
    try:
        inv_trap = np.linalg.inv(A_trap)
    except np.linalg.LinAlgError:
        inv_trap = np.linalg.pinv(A_trap)

    T_tri  = N_tri_np.dot(inv_tri)
    T_trap = N_trap_np.dot(inv_trap)

    r_tri = np.sum(T_tri, axis=1); c_tri = np.sum(T_tri, axis=0); rc_tri = r_tri + c_tri
    r_trap = np.sum(T_trap, axis=1); c_trap = np.sum(T_trap, axis=0); rc_trap = r_trap + c_trap

    w_tri  = (rc_tri / float(np.sum(rc_tri))).tolist()
    w_trap = (rc_trap / float(np.sum(rc_trap))).tolist()
 
    extras = {
        "T_tri": T_tri.tolist(), "T_trap": T_trap.tolist(),
        "rc_tri": rc_tri.tolist(), "rm_tri": (r_tri - c_tri).tolist(),
        "rc_trap": rc_trap.tolist(), "rm_trap": (r_trap - c_trap).tolist(),
        "x_c_tri": float(np.mean(rc_tri)), "y_c_tri": float(np.mean(r_tri - c_tri)),
        "x_c_trap": float(np.mean(rc_trap)), "y_c_trap": float(np.mean(r_trap - c_trap)),
        "avg_T_tri": float(np.mean(T_tri)), "avg_T_trap": float(np.mean(T_trap)),
    }
    return w_tri, w_trap, extras

@app.route('/fuzzy_dematel/<string:id>')
def fuzzy_dematel(id):
    build_config_params(configuration_file)
    collection = mu.open_mongo_connection(config['mongo']['thor'])
    thor = mu.get_objects(collection, ObjectId(id))
    criterias = thor.get('criterias', [])
    decisors  = thor.get('decisors', [])
    d_index = int(request.args.get('dm', 1))
    thor['assignment_method_selected'] = "Fuzzy Dematel"
    data_DEMATEL = [
        ("No influence",  "0"),
        ("Very Low influence", "1"),
        ("Low influence", "2"),
        ("High influence", "3"),
        ("Very high influence", "4"),
    ]
    mu.update(ObjectId(id), thor, collection)
    return render_template(
        'fuzzy_dematel.html',
        id=id,
        d_index=d_index,
        criterias=criterias,
        decisors=decisors,
        data_DEMATEL=data_DEMATEL,
        title="Fuzzy Dematel"
    )

def _safe_decisor_key(name, d_index: int) -> str:
    if name is None:
        return f"Decision Maker {d_index}"
    s = str(name).strip()
    return s or f"Decision Maker {d_index}"

@app.route('/save_fuzzy_dematel/<string:id>', methods=['POST'])
def save_fuzzy_dematel(id):
    build_config_params(configuration_file)
    collection = mu.open_mongo_connection(config['mongo']['thor'])
    thor = mu.get_objects(collection, ObjectId(id))
    criterias = thor.get('criterias', [])
    decisors  = thor.get('decisors', [])
    n = len(criterias)
    d_index = int(request.form.get('d_index', 1))
    dm_name = decisors[d_index - 1] if decisors and 0 <= d_index-1 < len(decisors) else None
    current_decisor = (str(dm_name).strip() or f"Decision Maker {d_index}") if dm_name is not None else f"Decision Maker {d_index}"

    tri = np.zeros((n, n, 3), dtype=float)
    trap = np.zeros((n, n, 4), dtype=float)
    errors = []
    for i in range(n):
        for j in range(n):
            if i == j:
                tri[i, j]  = [0.0, 0.0, 0.0]      
                trap[i, j] = TRAP_DICT[0]         
            else:
                field = f"influence_{i+1}_{j+1}_{d_index}"
                raw = request.form.get(field, "").strip()
                if raw not in DEMATEL_ALLOWED:
                    errors.append((field, raw))
                else:
                    k = int(raw)
                    tri[i, j]  = TRI_DICT[k]
                    trap[i, j] = TRAP_DICT[k]

    if errors:
        flash("Há valores inválidos de influência. Use apenas 0–4.", "danger")
        return redirect(url_for('fuzzy_dematel', id=id, dm=d_index))

    tri_store  = {str(k): v for k, v in (thor.get('dematel_tri_matrices')  or {}).items()}
    trap_store = {str(k): v for k, v in (thor.get('dematel_trap_matrices') or {}).items()}
    tri_store[current_decisor]  = tri.tolist()
    trap_store[current_decisor] = trap.tolist()
    thor['dematel_tri_matrices']  = tri_store
    thor['dematel_trap_matrices'] = trap_store
    mu.update(ObjectId(id), thor, collection)

    if d_index < len(decisors):
        return redirect(url_for('fuzzy_dematel', id=id, dm=d_index+1))

    tri_list  = [np.array(m) for m in thor['dematel_tri_matrices'].values()]
    trap_list = [np.array(m) for m in thor['dematel_trap_matrices'].values()]
 
    w_tri, w_trap, extras = _compute_dematel_weights_exact(tri_list, trap_list)


    w_tri_rounded  = [round(x, 3) for x in w_tri]
    w_trap_rounded = [round(x, 3) for x in w_trap]

    thor['fuzzy_weights'] = {'triangular': w_tri_rounded, 'trapezoidal': w_trap_rounded}
    thor['dematel_extras'] = extras
    mu.update(ObjectId(id), thor, collection)
    return redirect(url_for('fuzzy_dematel_results', id=id))

def _classify_quadrants(rc: np.ndarray, rm: np.ndarray, x_c: float, y_c: float) -> list[str]:
    labels = []
    for x, y in zip(rc, rm):
        if x >= x_c and y >= y_c:
            labels.append("High prominence • High relationship")
        elif x >= x_c and y < y_c:
            labels.append("High prominence • Low relationship")
        elif x < x_c and y >= y_c:
            labels.append("Low prominence • High relationship")
        else:
            labels.append("Low prominence • Low relationship")
    return labels

def _pair_with_defaults(names: list[str], labels: list[str], default: str = "—") -> list[tuple[str, str]]:
    m = min(len(names), len(labels))
    pairs = list(zip(names[:m], labels[:m]))
    if len(names) > m:
        pairs.extend((n, default) for n in names[m:])
    return pairs

@app.route('/fuzzy_dematel_results/<string:id>')
def fuzzy_dematel_results(id):
    build_config_params(configuration_file)
    collection = mu.open_mongo_connection(config['mongo']['thor'])
    thor = mu.get_objects(collection, ObjectId(id))
    criterias = thor.get('criterias', [])
    weights = thor.get('fuzzy_weights', {})
    peso = weights.get('triangular', [])
    pesotrap = weights.get('trapezoidal', [])
    extras = thor.get('dematel_extras', {}) or {}

    dematel_pairs_tri, dematel_pairs_trap = [], []
    quadrants_tri, quadrants_trap = [], []
    try:
        T_tri   = np.array(extras.get('T_tri', []), dtype=float)
        T_trap  = np.array(extras.get('T_trap', []), dtype=float)
        avgTtri  = float(extras.get('avg_T_tri', 0.0))
        avgTtrap = float(extras.get('avg_T_trap', 0.0))

        if T_tri.size:
            for i in range(T_tri.shape[0]):
                for j in range(T_tri.shape[1]):
                    if T_tri[i, j] > avgTtri:
                        dematel_pairs_tri.append((i, j))
        if T_trap.size:
            for i in range(T_trap.shape[0]):
                for j in range(T_trap.shape[1]):
                    if T_trap[i, j] > avgTtrap:
                        dematel_pairs_trap.append((i, j))

        rc_tri  = np.array(extras.get('rc_tri', []), dtype=float)
        rm_tri  = np.array(extras.get('rm_tri', []), dtype=float)
        rc_trap = np.array(extras.get('rc_trap', []), dtype=float)
        rm_trap = np.array(extras.get('rm_trap', []), dtype=float)
        x_c_tri = float(extras.get('x_c_tri', 0.0))
        y_c_tri = float(extras.get('y_c_tri', 0.0))
        x_c_trap = float(extras.get('x_c_trap', 0.0))
        y_c_trap = float(extras.get('y_c_trap', 0.0))

        if rc_tri.size and rm_tri.size:
            quadrants_tri = _classify_quadrants(rc_tri, rm_tri, x_c_tri, y_c_tri)
        if rc_trap.size and rm_trap.size:
            quadrants_trap = _classify_quadrants(rc_trap, rm_trap, x_c_trap, y_c_trap)
    except Exception:
        pass

    quad_pairs_tri  = _pair_with_defaults(criterias, quadrants_tri, default="(sem classificação)")
    quad_pairs_trap = _pair_with_defaults(criterias, quadrants_trap, default="(sem classificação)")

    return render_template(
        'fuzzy_dematel_results.html',
        id=id,
        criterias=criterias,
        peso=peso,
        pesotrap=pesotrap,
        dematel_extras=extras,
        dematel_pairs_tri=dematel_pairs_tri,
        dematel_pairs_trap=dematel_pairs_trap,
        quad_pairs_tri=quad_pairs_tri,
        quad_pairs_trap=quad_pairs_trap,
        title="Fuzzy Dematel Results"
    )

@app.route('/fuzzy_dematel_results_submit/<string:id>', methods=['POST'])
def fuzzy_dematel_results_submit(id):
    build_config_params(configuration_file)
    collection = mu.open_mongo_connection(config['mongo']['thor'])
    thor = mu.get_objects(collection, ObjectId(id))

    fw = thor.get('fuzzy_weights')
    if not fw:
        flash("Não há pesos fuzzy calculados para DEMATEL.", "danger")
        return redirect(url_for('fuzzy_dematel_results', id=id))

    thor['peso_triangular']  = fw.get('triangular', [])
    thor['peso_trapezoidal'] = fw.get('trapezoidal', [])

    num_decisors = len(thor.get('decisors', []))
    thor['pesofim'] = [thor['peso_triangular'] for _ in range(num_decisors)]
    thor['peso']    = thor['peso_triangular']

    mu.update(ObjectId(id), thor, collection)

    return render_template(
        'matrix.html',
        id=id,
        title='Matrix',
        peso=thor['peso'],
        pesotrap=thor['peso_trapezoidal'],
        pesofims=thor['pesofim'],
        criterias=thor.get('criterias', []),
        alternatives=thor.get('alternatives', []),
        decisors=thor.get('decisors', []),
        thor=thor
    )

@app.route('/escala/<string:id>', methods=['GET','POST'])
def escala(id):
    collection = None
    while collection is None:
        build_config_params(configuration_file)
        collection = mu.open_mongo_connection(config['mongo']['thor'])
    thor = mu.get_objects(collection, ObjectId(id))
    thor['assignment_method_selected'] = 2
    cri = len(thor['criterias'])
    if len(request.form) > 0:
        for i in range(1, len(thor['decisors']) + 1):
            pesom=[1]
            for j in range(cri-1):
                pesom.append(0)
            temp = []
            for j in range(cri-1):
                d = j + 1
                pref = float(request.form[f'decisor-s-{i}-{d}'])
                pesom[d]=pesom[j]+pref
                temp.append("What's your preferences of " + thor['questions'][j] + " : " + str(pref))
            padrao=min(pesom)
            if padrao<=0:
                for j in range(cri):
                    pesom[j]+=(1-padrao)
            thor['pesomList'].append(pesom)
            thor['answer'].append(temp)
        thor['pesom'] = thor['pesomList'][0]
    else:
        thor['marc'] +=1
        thor['indexCriMarc'] = 0
    mu.update(ObjectId(id), thor, collection)
    return render_template ('weight_continue.html',
                                id=id,
                                dindex=thor['indexDecisor'] + 1,
                                title='Accept this weight',
                                pesom=thor['pesom'])

@app.route('/razao/<string:id>', methods=['GET','POST'])
def razao(id):
    collection = None
    while collection is None:
        build_config_params(configuration_file)
        collection = mu.open_mongo_connection(config['mongo']['thor'])
    thor = mu.get_objects(collection, ObjectId(id))
    thor['assignment_method_selected'] = 3
    cri = len(thor['criterias'])
    if len(request.form) > 0:
        for i in range(1, len(thor['decisors']) + 1):
            pesom=[1]
            for j in range(cri-1):
                pesom.append(0)
            temp = []
            for j in range(cri-1):
                d = j + 1
                pref = request.form[f'decisor-r-{i}-{d}']
                pref = float(pref)
                pesom[j+1]=pesom[j]*pref
                temp.append("What's your preferences of " + thor['questions'][j] + " : " + str(pref))
            thor['pesomList'].append(pesom)
            thor['answer'].append(temp)
        thor['pesom'] = thor['pesomList'][0]
    else:
        thor['marc'] +=1
        thor['indexCriMarc'] = 0
    mu.update(ObjectId(id), thor, collection)
    return render_template ('weight_continue.html',
                                id=id,
                                dindex=thor['indexDecisor'] + 1,
                                title='Accept this weight',
                                pesom=thor['pesom'])

@app.route('/weightregarding/<string:id>', methods=['GET','POST'])
def weightregarding(id):
    collection = None
    while collection is None:
        build_config_params(configuration_file)
        collection = mu.open_mongo_connection(config['mongo']['thor'])
    thor = mu.get_objects(collection, ObjectId(id))
    cri = len(thor['criterias'])
    if request.method == 'POST':
        thor['answer'][thor['indexCriMarc'] - 1][-1] = thor['answer'][thor['indexCriMarc'] - 1][-1] + " : " + str(float(request.form['r']))

        if thor['assignment_method_selected'] == 3:
            thor['pesom'][thor['indexCriMarc']+thor['marc']]= float(request.form['r'])
        else:
            thor['pesom'][thor['indexCriMarc']+thor['marc']]=thor['pesom'][thor['indexCriMarc']]+float(request.form['r'])
        thor['indexCriMarc'] += 1

    if thor['marc']==cri-1 and "between" in request.referrer:
        if thor['indexDecisor'] < len(thor['decisors']):
            thor['marc'] = 1
            thor['pesomList'][thor['indexDecisor']] = thor['pesom']
            thor['indexDecisor'] += 1
            thor['pesofim'].append(thor['pesom'])
            if thor['indexDecisor'] == len(thor['decisors']):
                mu.update(ObjectId(id), thor, collection)
                return redirect(url_for('matrix', id=id))
            thor['pesom'] = thor['pesomList'][thor['indexDecisor']]
            mu.update(ObjectId(id), thor, collection)
            if thor['assignment_method_selected'] == 3:
                return redirect(url_for('razao', id=id))
            else:
                return redirect(url_for('escala', id=id))
        mu.update(ObjectId(id), thor, collection)
        return redirect(url_for('matrix', id=id))

    if "razao" not in request.referrer or "escala" not in request.referrer :
        if thor['indexCriMarc'] >= cri-thor['marc']:
            mu.update(ObjectId(id), thor, collection)
            if thor['assignment_method_selected'] == 3:
                return redirect(url_for('razao', id=id))
            else:
                return redirect(url_for('escala', id=id))
    q = thor['criterias'][thor['indexCriMarc']+thor['marc']] +' regarding '+ thor['criterias'][thor['indexCriMarc']]
    thor['answer'][thor['indexCriMarc'] - 1].append("What's your preferences of " + q)
    mu.update(ObjectId(id), thor, collection)
    return render_template ('weight_regarding.html',
                                id=id,
                                title='Regarding Weight',
                                questions=q,
                                dIndex=thor['indexDecisor'] + 1)

@app.route('/weightbetween/<string:id>', methods=['GET','POST'])
def weightbetween(id):
    collection = None
    while collection is None:
        build_config_params(configuration_file)
        collection = mu.open_mongo_connection(config['mongo']['thor'])
    thor = mu.get_objects(collection, ObjectId(id))
    input_value = float(request.form['r'])
    thor['answer'][thor['indexCriMarc'] - 1][-1] = thor['answer'][thor['indexCriMarc'] - 1][-1] + " : " + str(input_value)
    if thor['assignment_method_selected'] == 3:
        pref = input_value*thor['pesom'][thor['indexCriMarc']]
        mi = min(pref,thor['pesom'][thor['indexCriMarc']+thor['marc']])
        ma = max(pref,thor['pesom'][thor['indexCriMarc']+thor['marc']])
    else:
        pref = input_value
        mi = pref
        ma = thor['pesom'][thor['indexCriMarc']+thor['marc']]-thor['pesom'][thor['indexCriMarc']]
    q = "choose a value between " + str(mi) + " and " + str(ma) + " to " + thor['criterias'][thor['indexCriMarc']+thor['marc']]
    thor['answer'][thor['indexCriMarc'] - 1].append("What's your preferences of " + q)
    mu.update(ObjectId(id), thor, collection)
    return render_template('weight_between.html',
                                id=id,
                                mi=mi,
                                ma=ma,
                                title='Weight between',
                                questions=q,
                                dIndex=thor['indexDecisor'] +1)

@app.route('/matrix/<string:id>', methods=['GET', 'POST'])
def matrix(id):
    collection = None
    while collection is None:
        build_config_params(configuration_file)
        collection = mu.open_mongo_connection(config['mongo']['thor'])
    thor = mu.get_objects(collection, ObjectId(id))
  
    if thor['indexDecisor'] < len(thor['decisors']) and len(thor['pesomList']) != 0:
            thor['marc'] = 1
            thor['pesomList'][thor['indexDecisor']] = thor['pesom']
            thor['indexDecisor'] += 1
            thor['pesofim'].append(thor['pesom'])
            if thor['indexDecisor'] == len(thor['decisors']):
                mu.update(ObjectId(id), thor, collection)
                return redirect(url_for('matrix', id=id))
            thor['pesom'] = thor['pesomList'][thor['indexDecisor']]
            mu.update(ObjectId(id), thor, collection)
            if thor['assignment_method_selected'] == 3:
                return redirect(url_for('razao', id=id))
            else:
                return redirect(url_for('escala', id=id))
    pesom = thor['pesom']
    pesofim = thor['pesofim']
    cri = len(thor['criterias'])
    peso = [ 0 for i in range(cri)]
    if thor['assignment_method_selected'] == 3:
        for i in range(len(thor['decisors'])):
            norm=max(pesofim[i])
            for j in range(cri):
            
                if i==0:
                
                    peso[j]+=(pesofim[i][j]/norm)
                else:
                    
                    peso[j]*=(pesofim[i][j]/norm)
        if len(thor['decisors']) > 1:
            for i in range(cri):
                peso[i]=round(peso[i]**(1/2), 5)
        else:
            peso = [round(peso[i],5) for i in range(cri)]
    elif thor['assignment_method_selected'] == 2:
        for i in range(1, len(thor['decisors']) + 1):
            norm=max(pesofim[i-1])
            for j in range(cri):
                peso[j]+=round((pesofim[i-1][j]/norm), 5)
    else:
        thor['assignment_method_selected'] = 1
        pesofim = []
        cri = len(thor['criterias'])
        for i in range(1, len(thor['decisors']) + 1):
            pesofim.append([float(request.form[f'decisor-{i}-{j}']) for j in range(1, len(thor['criterias']) + 1)])
        for i in range(len(thor['decisors'])):
            norm=max(pesofim[i])
            for j in range(cri):
                peso[j]+=round((pesofim[i][j]/norm), 5)

    thor['peso'] = peso
    # >>> FIX: só acessar trapezoidal se for THOR II FUZZY (selected_method == 3) e o campo existir
    selected_method = thor.get('selected_method')
    pesotrap = None
    if selected_method == 3:
        pesotrap = thor.get('peso_trapezoidal') or thor.get('fuzzy_weights', {}).get('trapezoidal')
        # garante consistência no documento se existir
        if pesotrap is not None:
            thor['peso_trapezoidal'] = pesotrap
    # <<<
    thor['pesofim'] = pesofim
    mu.update(ObjectId(id), thor, collection)
    return render_template('matrix.html',
                           id=id,
                           title='Matrix',
                           peso=peso,
                           pesotrap=pesotrap,
                           pesofims=pesofim,
                           criterias=thor['criterias'],
                           alternatives=thor['alternatives'],
                           decisors=thor['decisors'],
                           thor=thor)

def build_ranking_with_separators(scores, alternatives):
    idx_sorted = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked = [alternatives[idx_sorted[0]]]
    for k in range(len(idx_sorted) - 1):
        i = idx_sorted[k]
        j = idx_sorted[k + 1]
        ranked.append(">" if scores[i] > scores[j] else "=")
        ranked.append(alternatives[idx_sorted[k + 1]])
    return ranked

def run_single_result(thorBd, peso_override, tag_label):
    thor = Thor()
    thor.alternatives = thorBd['alternatives']
    thor.decisors = thorBd['decisors']
    thor.criterias = thorBd['criterias']
    thor.selected_method = thorBd['selected_method']    
    thor.peso = list(map(float, peso_override))         
    thor.pesofim = thorBd['pesofim']
    thor.assignment_method_selected = thorBd['assignment_method_selected']
    thor.answer = thorBd['answer']

    # flags de formulário
    user_pertinence = (request.form['pertinence'] == '1')
    usartca_flag = (request.form['tca'] == '1')  
    thor.user_pertinence = user_pertinence
    thor.usartca = usartca_flag

    # PDF base
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font('Arial', '', 9)
    answer = (
    'FUZZY THOR II' if thor.selected_method == 3
    else 'THOR II' if thor.selected_method == 2
    else 'THOR I'
    )
    if thor.selected_method == 3:
        pdf.cell(0, 5, f'Run: {tag_label}', 0, 1)
    pdf.cell(0, 5, f'Method selected : ' + answer, 0, 1)
    pdf.cell(0, 5, f"Number of alternatives : {len(thor.alternatives)}", 0, 1)
    pdf.cell(0, 5, f"Number of criterias : {len(thor.criterias)}", 0, 1)
    pdf.cell(0, 5, f"Number of decisors : {len(thor.decisors)}", 0, 1)
    pdf.cell(0, 5, '', 0, 1)

    pdf.cell(0, 5, 'Alternatives name: ', 0, 1)
    pdf.cell(0, 5, f"{' - '.join(thor.alternatives)}", 0, 1)
    pdf.cell(0, 5, 'Criterias name :', 0, 1)
    pdf.cell(0, 5, f"{' - '.join(thor.criterias)}", 0, 1)

    pdf.cell(0, 5, '', 0, 1)
    pdf.cell(0, 5, 'Weights section:', 0, 1)
    if thor.assignment_method_selected == 3:
        pdf.cell(0, 5, 'Assignment method selected : Reason Scale', 0, 1)
    elif thor.assignment_method_selected == 2:
        pdf.cell(0, 5, 'Assignment method selected : Interval Scale', 0, 1)
    elif thor.assignment_method_selected == 1:
        pdf.cell(0, 5, 'Assignment method selected : Direct Assignment', 0, 1)
    else:
        pdf.cell(0, 5, f'Assignment method selected : {thor.assignment_method_selected}', 0, 1)

    for i in range(len(thor.decisors)):
        text = f"Decisor {i+1} :"
        if thor.answer and thor.answer[i]:
            for j in thor.answer[i]:
                pdf.cell(0, 5, text + j, 0, 1)
        lista= ' - '.join(str(x) for x in thor.pesofim[i])
        pdf.cell(0, 5, text + lista, 0, 1)

    pdf.cell(0, 5, f"Final weights : {' - '.join(str(x) for x in thor.peso)}", 0, 1)
    pdf.cell(0, 5, '', 0, 1)

    # -------------------------
    # Variáveis e leitura form
    # -------------------------
    alternativas = thor.alternatives
    criterios    = thor.criterias
    cri = len(criterios)
    alt = len(alternativas)

    q = [float(request.form[f'value-q-{i}']) for i in range(1, cri + 1)]
    p = [float(request.form[f'value-p-{i}']) for i in range(1, cri + 1)]
    d = [float(request.form[f'value-d-{i}']) for i in range(1, cri + 1)]

    matriz = [[0.0]*cri for _ in range(alt)]
    for i in range(1, alt + 1):
        for j in range(1, cri + 1):
            matriz[i-1][j-1]=float(request.form[f'alternative-value-{i}-{j}'])

    # pertinência
    pertinencia=[]; pertinenciatca=[]
    pertinencia2 = [[0.0]*cri for _ in range(alt)]
    pertinencia2tca = [[0.0]*cri for _ in range(alt)]
    if user_pertinence:
        pdf.cell(0, 5, "Do you would like to use pertinence: Yes", 0, 1)
        for i in range(1, cri+1):
            v = float(request.form[f'value-matrix-c-{i}'])
            pertinencia.append(v); pertinenciatca.append(v)
        pdf.cell(0, 5, f"Pertinence weights : {' - '.join(str(x) for x in pertinencia)}", 0, 1)
        pdf.cell(0, 5, f"Pertinence matrix section:", 0, 1)
        for i in range(1, alt+1):
            for j in range(1, cri+1):
                val = float(request.form[f'value-matrix-p-{i}-{j}'])
                pertinencia2[i-1][j-1] = val
                pertinencia2tca[i-1][j-1] = val
            pdf.cell(0, 5, f"{' - '.join(str(x) for x in pertinencia2[i-1])}", 0, 1)
        pdf.cell(0, 5, '', 0, 1)
    else:
        pdf.cell(0, 5, "Do you would like to use pertinence: No", 0, 1)
        pertinencia = [1.0]*cri
        pertinenciatca = [1.0]*cri
        for i in range(alt):
            for j in range(cri):
                pertinencia2[i][j]=1.0
                pertinencia2tca[i][j]=1.0

    if thor.usartca:
        pdf.cell(0, 5, "Do you would like to use TCA: Yes", 0, 1)
    else:
        pdf.cell(0, 5, "Do you would like to use TCA: No", 0, 1)
    pdf.cell(0, 5, '', 0, 1)
    pdf.cell(0, 5, f"Matrix section:", 0, 1)
    for i in range(1, len(thor.alternatives) + 1):
        for j in range(1, len(thor.criterias) + 1):
            matriz[i-1][j-1]=float(request.form[f'alternative-value-{i}-{j}'])
        pdf.cell(0, 5, f"{' - '.join(str(x) for x in matriz[i-1])}", 0, 1)
    pdf.cell(0, 5, '', 0, 1)
    pdf.cell(0, 5, f"P : {' - '.join(str(x) for x in p)}", 0, 1)
    pdf.cell(0, 5, f"Q : {' - '.join(str(x) for x in q)}", 0, 1)
    pdf.cell(0, 5, f"D : {' - '.join(str(x) for x in d)}", 0, 1)
    pdf.cell(0, 5, '', 0, 1)
    # médias (TCA nebulosa)
    medtcan=[]
    media1 = round(Utils.media(pertinencia), 4)
    medtcan.append(media1)
    for i in range(alt):
        media1 = round(Utils.media(pertinencia2[i]), 4)
        medtcan.append(media1)
    for i in range(1, alt+1):
        media1 = round(((medtcan[0] + medtcan[i]) / 2), 4)
        medtcan.append(media1)

    # --------------------------------------
    # S1/S2/S3 
    # --------------------------------------
    matrizs1 = [[0.0]*alt for _ in range(alt)]
    matrizs2 = [[0.0]*alt for _ in range(alt)]
    matrizs3 = [[0.0]*alt for _ in range(alt)]
    peso   = thor.peso[:]     
    peso2  = thor.peso[:]
    peso4  = thor.peso[:]
    originals1=[]; originals2=[]; originals3=[]
    cris1=[];cris2=[];cris3=[]; cristotal=[]

    def fill_pairwise(matriz_s, use_thor2):
        for i in range(alt):
            for j in range(alt):
                if i<j:
                    b=[];c=[];e=[];f=[];g=[];h=[]
                    for k in range(cri):
                        x=Utils.compara(matriz[i][k],matriz[j][k],p,q,k)
                        y=Utils.dif(matriz[i][k],matriz[j][k])
                        w=Utils.compara(matriz[j][k],matriz[i][k],p,q,k)
                        z=Utils.dif(matriz[j][k],matriz[i][k])
                        v=Utils.ind(pertinencia[k],pertinencia2[i][k],pertinencia2[j][k])
                        t=Utils.ind(pertinencia[k],pertinencia2[j][k],pertinencia2[i][k])
                        b.append(y); c.append(x); e.append(w); f.append(z); g.append(v); h.append(t)
                    if not use_thor2:
                        if matriz_s is matrizs1:
                            cond=Utils.s1(c,b,g,peso,cri); cond2=Utils.s1(e,f,h,peso,cri); disc=Utils.discordancias1
                        elif matriz_s is matrizs2:
                            cond=Utils.s2(c,b,g,peso,cri); cond2=Utils.s2(e,f,h,peso,cri); disc=Utils.discordancias2
                        else:
                            cond=Utils.s3(c,b,g,peso,cri); cond2=Utils.s3(e,f,h,peso,cri); disc=Utils.discordancias3
                        if cond=="domina" and cond2=="domina":
                            check1=disc(b,c,g,d,peso,cri); check2=disc(f,e,h,d,peso,cri)
                            if check1==0.5 or check2==0.5:
                                matriz_s[i][j]=0.5; matriz_s[j][i]=0.5
                            else:
                                matriz_s[i][j]=round(disc(b,c,g,d,peso,cri),3)
                                matriz_s[j][i]=round(disc(f,e,h,d,peso,cri),3)
                        elif cond=="domina" and cond2!="domina":
                            ms=disc(b,c,g,d,peso,cri)
                            matriz_s[i][j]=round(ms,3) if ms!=0.5 else 0.5
                            matriz_s[j][i]=0 if ms!=0.5 else 0.5
                        elif cond!="domina" and cond2=="domina":
                            ms=disc(f,e,h,d,peso,cri)
                            matriz_s[i][j]=0 if ms!=0.5 else 0.5
                            matriz_s[j][i]=round(ms,3) if ms!=0.5 else 0.5
                        else:
                            matriz_s[i][j]=0.5; matriz_s[j][i]=0.5
                    else:
                        if matriz_s is matrizs1:
                            cond=Utils.s1T2(c,b,g,p,q,peso,cri); cond2=Utils.s1T2(e,f,h,p,q,peso,cri); disc=Utils.discordancias1T2
                        elif matriz_s is matrizs2:
                            cond=Utils.s2T2(c,b,g,p,q,peso,cri); cond2=Utils.s2T2(e,f,h,p,q,peso,cri); disc=Utils.discordancias2T2
                        else:
                            cond=Utils.s3T2(c,b,g,p,q,peso,cri); cond2=Utils.s3T2(e,f,h,p,q,peso,cri); disc=Utils.discordancias3T2
                        if cond=="domina" and cond2=="domina":
                            check1=disc(b,c,g,d,p,q,peso,cri); check2=disc(f,e,h,d,p,q,peso,cri)
                            if check1==0.5 or check2==0.5:
                                matriz_s[i][j]=0.5; matriz_s[j][i]=0.5
                            else:
                                matriz_s[i][j]=round(disc(b,c,g,d,p,q,peso,cri),3)
                                matriz_s[j][i]=round(disc(f,e,h,d,p,q,peso,cri),3)
                        elif cond=="domina" and cond2!="domina":
                            ms=disc(b,c,g,d,p,q,peso,cri)
                            matriz_s[i][j]=round(ms,3) if ms!=0.5 else 0.5
                            matriz_s[j][i]=0 if ms!=0.5 else 0.5
                        elif cond!="domina" and cond2=="domina":
                            ms=disc(f,e,h,d,p,q,peso,cri)
                            matriz_s[i][j]=0 if ms!=0.5 else 0.5
                            matriz_s[j][i]=round(ms,3) if ms!=0.5 else 0.5
                        else:
                            matriz_s[i][j]=0.5; matriz_s[j][i]=0.5

    use_thor2 = (thor.selected_method == 2) or (thor.selected_method == 3)
    
    # S1
    fill_pairwise(matrizs1, use_thor2)
    rs1 = [round(sum(matrizs1[i][j] for j in range(alt)),3) for i in range(alt)]
    ranked = build_ranking_with_separators(rs1, alternativas)
    originals1 = ranked[:]
    s1 = Result()
    s1.S_result  = [" ".join([alternativas[row], "-"] + [str(matrizs1[row][col]) for col in range(alt)]) for row in range(alt)]
    s1.somatorio = [" ".join([alternativas[row], "= ", str(rs1[row])]) for row in range(alt)]
    s1.original  = " ".join(ranked + [" - Original."])
    thor.result.append(s1)

    # S2
    fill_pairwise(matrizs2, use_thor2)
    rs2 = [round(sum(matrizs2[i][j] for j in range(alt)),3) for i in range(alt)]
    ranked = build_ranking_with_separators(rs2, alternativas)
    originals2 = ranked[:]
    s2 = Result()
    s2.S_result  = [" ".join([alternativas[row], "-"] + [str(matrizs2[row][col]) for col in range(alt)]) for row in range(alt)]
    s2.somatorio = [" ".join([alternativas[row], "= ", str(rs2[row])]) for row in range(alt)]
    s2.original  = " ".join(ranked + [" - Original."])
    thor.result.append(s2)

    # S3
    fill_pairwise(matrizs3, use_thor2)
    rs3 = [round(sum(matrizs3[i][j] for j in range(alt)),3) for i in range(alt)]
    ranked = build_ranking_with_separators(rs3, alternativas)
    originals3 = ranked[:]
    s3 = Result()
    s3.S_result  = [" ".join([alternativas[row], "-"] + [str(matrizs3[row][col]) for col in range(alt)]) for row in range(alt)]
    s3.somatorio = [" ".join([alternativas[row], "= ", str(rs3[row])]) for row in range(alt)]
    s3.original  = " ".join(ranked + [" - Original."])
    thor.result.append(s3)
    for r, obj in enumerate(thor.result, start=1):
        pdf.cell(0, 5, f"Result S{r} - {tag_label}", 0, 1)
        for line in obj.S_result:
            pdf.cell(0, 5, line, 0, 1)
        pdf.cell(0, 5, '', 0, 1)
        for line in obj.somatorio:
            pdf.cell(0, 5, line, 0, 1)
        pdf.cell(0, 5, obj.original, 0, 1)
        pdf.cell(0, 5, '', 0, 1)



    # ----------------------
    # TCA normal (se marcado)
    # ----------------------
    def tca_cycle(matriz_s, originals_list, title, cris_out, result_list_holder):
        nonlocal peso, peso2
        peso = peso4[:]; peso2 = peso4[:]; usados=[]
        for _ in range(cri):
            # remove o menor peso >0 ainda não usado
            cand = [(j, peso2[j]) for j in range(cri) if peso2[j] > 0 and j not in usados]
            if not cand: break
            idx = min(cand, key=lambda t: t[1])[0]
            usados.append(idx)
            peso[idx]=0; peso2[idx]=0

            # recalcula matriz_s com peso zerado
            fill_pairwise(matriz_s, use_thor2)
            if matriz_s is matrizs1:
                r = [round(sum(matrizs1[i][j] for j in range(alt)),3) for i in range(alt)]
                mref = matrizs1
            elif matriz_s is matrizs2:
                r = [round(sum(matrizs2[i][j] for j in range(alt)),3) for i in range(alt)]
                mref = matrizs2
            else:
                r = [round(sum(matrizs3[i][j] for j in range(alt)),3) for i in range(alt)]
                mref = matrizs3
            ranked_tca = build_ranking_with_separators(r, alternativas)

            rt = ResultTca()
            rt.title = title
            rt.sub_title = f'Criteria analyzed {criterios[idx]}'
            rt.S_result = [" ".join([alternativas[row], "-"] + [str(mref[row][col]) for col in range(alt)]) for row in range(alt)]
            rt.somatorio = [" ".join([alternativas[row], "= ", str(r[row])]) for row in range(alt)]
            rt.original  = " ".join(ranked_tca + [" - Without the criteria."])
            rt.original2 = " ".join(originals_list + [" - Original."])
            result_list_holder.append(rt)

            if all(ranked_tca[k] == originals_list[k] for k in range(len(ranked_tca))):
                crit = criterios[idx]
                cris_out.append(crit)

            
            peso = peso4[:]; peso2 = peso4[:]

    if thor.usartca:
        tca_cycle(matrizs1, originals1, "S1 result", cris1, thor.result_tca_s1)
        tca_cycle(matrizs2, originals2, "S2 result", cris2, thor.result_tca_s2)
        tca_cycle(matrizs3, originals3, "S3 result", cris3, thor.result_tca_s3)
        thor.tca_s1_citerio_removed = "No criteria can be removed" if not cris1 else "Criteria can be removed : " + " - ".join(cris1)
        thor.tca_s2_citerio_removed = "No criteria can be removed" if not cris2 else "Criteria can be removed : " + " - ".join(cris2)
        thor.tca_s3_citerio_removed = "No criteria can be removed" if not cris3 else "Criteria can be removed : " + " - ".join(cris3)
    if thor.usartca:
        pdf.cell(0, 5, f"TCA Result section - {tag_label}:", 0, 1)
        for lst in (thor.result_tca_s1, thor.result_tca_s2, thor.result_tca_s3):
            for obj in lst:
                pdf.cell(0, 5, obj.title, 0, 1)
                pdf.cell(0, 5, obj.sub_title, 0, 1)
                for line in obj.S_result:
                    pdf.cell(0, 5, line, 0, 1)
                pdf.cell(0, 5, '', 0, 1)
                for line in obj.somatorio:
                    pdf.cell(0, 5, line, 0, 1)
                pdf.cell(0, 5, obj.original, 0, 1)
                if getattr(obj, 'original2', None):
                    pdf.cell(0, 5, obj.original2, 0, 1)
                pdf.cell(0, 5, '', 0, 1)
    # ----------------------
    # TCA Nebulosa (se marcados TCA e pertinência)
    # ----------------------
    tcan = None
    if thor.usartca and thor.user_pertinence:
        if not hasattr(thor, "result_tca_n") or thor.result_tca_n is None:
            thor.result_tca_n = []
        else:
            thor.result_tca_n.clear()

        # original
        tcan = TcaNebulosa()
        pdf.cell(0, 5, "Nebulosa TCA original result section:", 0, 1)
        head = [["Weights - "]+[str(pertinencia[col]) for col in range(cri)]]
        input_rows = [[" "]+[str(alternativas[row])]+["- "]+[str(pertinencia2[row][col]) for col in range(cri)] for row in range(alt)]
        medias = [["Average of weights :"]+[str(medtcan[0])]]
        mediaalt = [["Average of "]+[str(alternativas[row])]+["-"]+[str(medtcan[row+1])] for row in range(alt)]
        mediamedias = [["Average of weights with"]+[str(alternativas[row])]+["-"]+[str(medtcan[row+1+alt])] for row in range(alt)]
        tcan.head = [" ".join(head[r]) for r in range(len(head))]
        tcan.input_rows = [" ".join(input_rows[r]) for r in range(len(input_rows))]
        tcan.medias = [" ".join(medias[r]) for r in range(len(medias))]
        tcan.mediamedias = [" ".join(mediamedias[r]) for r in range(len(mediamedias))]
        tcan.mediaalt = [" ".join(mediaalt[r]) for r in range(len(mediaalt))]

        
        cristotal = list(dict.fromkeys(cris1 + cris2 + cris3))
        for i in tcan.head:
            pdf.cell(0, 5, i , 0, 1)
        pdf.cell(0, 5, '', 0, 1)
        for i in tcan.input_rows:
            pdf.cell(0, 5, i , 0, 1)
        pdf.cell(0, 5, '', 0, 1)
        for i in tcan.medias:
            pdf.cell(0, 5, i , 0, 1)
        pdf.cell(0, 5, '', 0, 1)
        for i in tcan.mediamedias:
            pdf.cell(0, 5, i , 0, 1)
        pdf.cell(0, 5, '', 0, 1)
        for i in tcan.mediaalt:
            pdf.cell(0, 5, i , 0, 1)
        pdf.cell(0, 5, '', 0, 1)
        tcaneb = []
        for crit_name in cristotal:
            neb_count = 0
            rtcan = []
            rt2 = []
            pos = criterios.index(crit_name)

            
            pertinencia.pop(pos)
            z = round(Utils.media(pertinencia), 4); rtcan.append(z)

            for k in range(alt):
                pertinencia2[k].pop(pos)
                z = round(Utils.media(pertinencia2[k]), 4); rtcan.append(z)
                z = round(((rtcan[k+1] + rtcan[0]) / 2), 4); rt2.append(z)

            # bloco calculado
            tcanCalc = TcaNebulosa()
            pdf.cell(0, 5, "Nebulosa TCA calculated result section:", 0, 1)
            header_calc = [['Removing criteria']+[str(crit_name)]]
            head_calc = [["Weights - "]+[str(pertinencia[col]) for col in range(len(pertinencia))]]
            input_rows_calc = [[" "]+[str(alternativas[row])]+["- "]+[str(pertinencia2[row][col]) for col in range(len(pertinencia))] for row in range(alt)]
            medias_calc = [["Average of weights :"]+[str(rtcan[0])]]
            mediaalt_calc = [["Average of "]+[str(alternativas[row])]+["-"]+[str(rtcan[row+1])] for row in range(alt)]
            mediamedias_calc = [["Average of weights with"]+[str(alternativas[row])]+["-"]+[str(rt2[row])] for row in range(alt)]

            tcanCalc.title = [" ".join(header_calc[r]) for r in range(len(header_calc))]
            tcanCalc.head = [" ".join(head_calc[r]) for r in range(len(head_calc))]
            tcanCalc.input_rows = [" ".join(input_rows_calc[r]) for r in range(len(input_rows_calc))]
            tcanCalc.medias = [" ".join(medias_calc[r]) for r in range(len(medias_calc))]
            tcanCalc.mediamedias = [" ".join(mediamedias_calc[r]) for r in range(len(mediamedias_calc))]
            tcanCalc.mediaalt = [" ".join(mediaalt_calc[r]) for r in range(len(mediaalt_calc))]

            thor.result_tca_n.append(tcanCalc)
            for i in tcanCalc.title:
                pdf.cell(0, 5, i , 0, 1)
            pdf.cell(0, 5, '', 0, 1)
            for i in tcanCalc.head:
                pdf.cell(0, 5, i , 0, 1)
            pdf.cell(0, 5, '', 0, 1)
            for i in tcanCalc.input_rows:
                pdf.cell(0, 5, i , 0, 1)
            pdf.cell(0, 5, '', 0, 1)
            for i in tcanCalc.medias:
                pdf.cell(0, 5, i , 0, 1)
            pdf.cell(0, 5, '', 0, 1)
            for i in tcanCalc.mediamedias:
                pdf.cell(0, 5, i , 0, 1)
            pdf.cell(0, 5, '', 0, 1)
            for i in tcanCalc.mediaalt:
                pdf.cell(0, 5, i , 0, 1)
            pdf.cell(0, 5, '', 0, 1)

            
            for k in range(alt):
                pertinencia2[k].insert(pos, pertinencia2tca[k][pos])
            pertinencia.insert(pos, pertinenciatca[pos])

            rtcan_full = rtcan + rt2
            if all(rtcan_full[j] >= medtcan[j] and rtcan_full != medtcan for j in range(len(rtcan_full))):
                tcaneb.append(crit_name)

        thor.removedTcaN = "By TCA nebulosa no criteria can be removed" if not tcaneb \
                           else "By tca nebulosa criteria can be removed : " + " - ".join(tcaneb)
        pdf.cell(0, 5, thor.removedTcaN, 0, 1)

  
    filename = 'static/' + datetime.now().strftime("%m_%d_%y_%H%M%S") + f'_{tag_label}.pdf'
    pdf.output(filename, 'F')

    # atacha metadado pra UI
    thor.tag = tag_label
    thor.file = filename
    return {"thor": thor, "tcan": tcan, "file": filename, "tag": tag_label}

@app.route('/result/<string:id>', methods=['POST'])
def result(id):
    # abre DB
    collection = None
    while collection is None:
        build_config_params(configuration_file)
        collection = mu.open_mongo_connection(config['mongo']['thor'])
    thorBd = mu.get_objects(collection, ObjectId(id))

    # pesos disponíveis
    fuzzy = (thorBd['selected_method'] == 3)
    fuzzy_tri = thorBd.get('fuzzy_weights', {}).get('triangular') or thorBd.get('peso') or thorBd.get('peso_triangular')
    fuzzy_trap = thorBd.get('fuzzy_weights', {}).get('trapezoidal') or thorBd.get('peso_trapezoidal')

    if fuzzy and fuzzy_trap:
        # roda duas vezes: TRI e TRAP
        tri_ctx  = run_single_result(thorBd, fuzzy_tri,  "Triangular")
        trap_ctx = run_single_result(thorBd, fuzzy_trap, "Trapezoidal")
        # manda lista para o template
        return render_template(
            'result.html',
            title="Result",
            thors=[tri_ctx["thor"], trap_ctx["thor"]],
            tcans=[tri_ctx["tcan"],  trap_ctx["tcan"]],
        )

    # caso não seja fuzzy (ou não tenha trap disponível), roda 1 vez só
    base_peso = thorBd.get('peso')
    single_ctx = run_single_result(thorBd, base_peso, "Default")
    return render_template('result.html', title="Result", thor=single_ctx["thor"], tcan=single_ctx["tcan"], file=single_ctx["file"])

