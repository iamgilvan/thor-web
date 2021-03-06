from flask import render_template, request, redirect, session, url_for
from app import app
from models.thor import *
from utils.utils import *
import json
from bson.objectid import ObjectId
from utils import mongo_utils as mu
from utils.pdf_utils import PDF
import os
from datetime import datetime

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
    questions =[]
    for j in range(cri-1):
        questions.append( thor['criterias'][j+1] +' regarding '+ thor['criterias'][j])
    thor['questions'] = questions
    mu.update(ObjectId(id), thor, collection)
    return render_template("weight.html",
                            questions=questions,
                            id=id,
                            title='Weight',
                            criterias=thor['criterias'],
                            alternatives=thor['alternatives'],
                            decisors=thor['decisors'])


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
    #se tiver mais decisor precisa redirecionar 
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
                #Mudança na normalização para média geométrica
                if i==0:
                    #Passa a somar apenas no primeiro valor
                    peso[j]+=(pesofim[i][j]/norm)
                else:
                    #Passa a multiplicar os seguintes
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
    thor['pesofim'] = pesofim
    mu.update(ObjectId(id), thor, collection)
    return render_template('matrix.html',
                           id=id,
                           title='Matrix',
                           peso=peso,
                           pesofims=pesofim,
                           criterias=thor['criterias'],
                           alternatives=thor['alternatives'],
                           decisors=thor['decisors'])


@app.route('/result/<string:id>', methods=['POST'])
def result(id):
    collection = None
    while collection is None:
        build_config_params(configuration_file)
        collection = mu.open_mongo_connection(config['mongo']['thor'])
    thorBd = mu.get_objects(collection, ObjectId(id))
    # obter valores do objeto antigo
    thor = Thor()
    thor.alternatives = thorBd['alternatives']
    thor.decisors = thorBd['decisors']
    thor.criterias = thorBd['criterias']
    thor.selected_method = thorBd['selected_method']
    thor.peso = thorBd['peso']
    thor.pesofim = thorBd['pesofim']
    thor.assignment_method_selected = thorBd['assignment_method_selected']
    thor.answer = thorBd['answer']

    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font('Arial', '', 9)
    answer = 'THOR II' if thor.selected_method == 2 else 'THOR I'
    pdf.cell(0, 5, f'Method selected : ' + answer, 0, 1)

    peso=[];peso2=[];peso3=[];peso4=[];ms1=0;ms2=0;ms3=0;p=[];q=[];d=[];t1=[];t2=[];t3=[];escolha=0;continuar=0;controle=0;testar=0;controlemaior=0;dic=0;controlador=0
    matrizs1=[];matrizs2=[];matrizs3=[];rs1=[];rs2=[];rs3=[];rs1o=[];rs2o=[];rs3o=[];pertinencia=[];pertinencia2=[];pertinenciatca=[];pertinencia2tca=[];indice=[];alternativas=[];alternativaso=[];criterios=[];cris1=[];cris2=[];cris3=[];cristotal=[];var=0;contador=0;originals1=[];originals2=[];originals3=[];medtcan=[];rtcan=[];tcaneb=[];neb=0;indice=0;tca1=0;tca2=0;tca3=0;f1=0;f2=0;f3=0;ver1=0;ver2=0;ver3=0;pos=0;pesofim=[];pesodec=[];pesom=[1];pesom2=[1];norm=0;ok=0

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
    else:
        pdf.cell(0, 5, 'Assignment method selected : Direct Assignment', 0, 1)

    for i in range(len(thor.decisors)):
        text = f"Decisor {i+1} :"
        if thor.answer and thor.answer[i]:
            for j in thor.answer[i]:
                pdf.cell(0, 5, text + j, 0, 1)
        lista= ' - '.join(str(x) for x in thor.pesofim[i])
        pdf.cell(0, 5, text + lista, 0, 1)

    pdf.cell(0, 5, f"Final weights : {' - '.join(str(x) for x in thor.peso)}", 0, 1)
    pdf.cell(0, 5, '', 0, 1)

    cri = len(thor.criterias)
    criterios = thor.criterias
    alt = len(thor.alternatives)
    alternativas = thor.alternatives
    peso2 = [ 0 for i in range(cri)]
    peso4 = [ 0 for i in range(cri)]
    for a in range(alt):
        linha=[]
        for b in range(alt):
            linha.append(0)
        matrizs1.append(linha)
    for a in range(alt):
        linha=[]
        for b in range(alt):
            linha.append(0)
        matrizs2.append(linha)
    for a in range(alt):
        linha=[]
        for b in range(alt):
            linha.append(0)
        matrizs3.append(linha)

    peso = thor.peso
    pesofim = thor.pesofim

    for i in range(cri):
        peso2[i]=peso[i]
        peso4[i]=peso[i]

    q = [float(request.form[f'value-q-{i}']) for i in range(1, len(thor.criterias) + 1)]
    p = [float(request.form[f'value-p-{i}']) for i in range(1, len(thor.criterias) + 1)]
    d = [float(request.form[f'value-d-{i}']) for i in range(1, len(thor.criterias) + 1)]

    matriz=[]
    for a in range(alt):
        linha=[]
        for b in range(cri):
            linha.append(0)
        matriz.append(linha)
    for a in range(alt):
        linha=[]
        for b in range(cri):
            linha.append(0)
        pertinencia2.append(linha)
    for a in range(alt):
        linha=[]
        for b in range(cri):
            linha.append(0)
        pertinencia2tca.append(linha)
    user_pertinence = True if request.form['pertinence'] == '1' else False
    usartca = 0 if request.form['tca'] == '1' else 1
    thor.user_pertinence = user_pertinence
    thor.usartca = True if usartca == 0 else False
    if user_pertinence:
        pdf.cell(0, 5, "Do you would like to use pertinence: Yes", 0, 1)
        #pegado os pesos da matriz de pertinencia
        for i in range(1, cri+1):
            pertinencia.append(float(request.form[f'value-matrix-c-{i}']))
            pertinenciatca.append(float(request.form[f'value-matrix-c-{i}']))
        pdf.cell(0, 5, f"Pertinence weights : {' - '.join(str(x) for x in pertinencia)}", 0, 1)
        #pegado a matriz de pertinencia
        pdf.cell(0, 5, f"Pertinence matrix section:", 0, 1)
        for i in range(1, alt+1):
            for j in range(1, cri+1):
                pertinencia2[i-1][j-1]=float(request.form[f'value-matrix-p-{i}-{j}'])
                pertinencia2tca[i-1][j-1]=float(request.form[f'value-matrix-p-{i}-{j}'])
            pdf.cell(0, 5, f"{' - '.join(str(x) for x in pertinencia2[i-1])}", 0, 1)
        pdf.cell(0, 5, '', 0, 1)
    else:
        pdf.cell(0, 5, "Do you would like to use pertinence: No", 0, 1)
        for i in range(cri):
            pertinencia.append(1)
            pertinenciatca.append(1)
        for i in range(alt):
            for j in range(cri):
                pertinencia2[i][j]=1
                pertinencia2tca[i][j]=1

    if thor.usartca:
        pdf.cell(0, 5, "Do you would like to use TCA: Yes", 0, 1)
    else:
        pdf.cell(0, 5, "Do you would like to use TCA: No", 0, 1)

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

    media1=round(Utils.media(pertinencia),4)
    medtcan.append(media1)
    for i in range(alt):
        media1=round(Utils.media(pertinencia2[i]),4)
        medtcan.append(media1)
    for i in range(1,alt+1):
        media1=round(((medtcan[0]+medtcan[i])/2),4)
        medtcan.append(media1)

    #S1
    c=[];b=[];e=[];f=[];g=[];h=[];menor=max(peso)
    while contador<1:
        c=[];b=[];e=[];f=[];g=[];h=[];x=0;y=0;w=0;z=0;v=0;t=0;rs1=[];rs1o=[];
        if contador!=0:
            menor=max(peso2)
            for j in range(cri):
                if peso2[j]<menor and peso2.index(menor) not in peso3 and peso2[j]>0:
                    menor=peso2[j]
            peso3.append(peso2.index(menor))
            peso2[peso2.index(menor)]=0
            peso[peso3[contador-1]]=0
        for i in range(alt):
            for j in range(alt):
                if(i<j):
                    for k in range(cri):
                        x=Utils.compara(matriz[i][k],matriz[j][k], p,q,k)
                        y=Utils.dif(matriz[i][k],matriz[j][k])
                        w=Utils.compara(matriz[j][k],matriz[i][k], p,q,k)
                        z=Utils.dif(matriz[j][k],matriz[i][k])
                        v=Utils.ind(pertinencia[k],pertinencia2[i][k],pertinencia2[j][k])
                        t=Utils.ind(pertinencia[k],pertinencia2[j][k],pertinencia2[i][k])
                        b.append(y)
                        c.append(x)
                        e.append(w)
                        f.append(z)
                        g.append(v)
                        h.append(t)
                    if thor.selected_method == 1:
                        if(Utils.s1(c,b,g, peso, cri)=="domina") and (Utils.s1(e,f,h, peso, cri)=="domina"):
                            check1 = Utils.discordancias1(b,c,g,d,peso, cri)
                            check2 = Utils.discordancias1(f,e,h,d,peso, cri)
                            if(check1==0.5) or (check2==0.5):
                                matrizs1[i][j]=0.5
                                matrizs1[j][i]=0.5
                            else:
                                matrizs1[i][j]=round(Utils.discordancias1(b,c,g,d,peso, cri),3)
                                matrizs1[j][i]=round(Utils.discordancias1(f,e,h,d,peso, cri),3)
                        elif(Utils.s1(c,b,g, peso, cri)=="domina") and (Utils.s1(e,f,h, peso, cri)!="domina"):
                            ms1 = Utils.discordancias1(b,c,g,d,peso, cri)
                            if(ms1 !=0.5):
                                matrizs1[i][j]=round(ms1,3)
                                matrizs1[j][i]=0
                            else:
                                matrizs1[i][j]=0.5
                                matrizs1[j][i]=0.5
                        elif(Utils.s1(c,b,g,peso, cri)!="domina") and (Utils.s1(e,f,h, peso, cri)=="domina"):
                            ms1 = Utils.discordancias1(f,e,h,d,peso, cri)
                            if(ms1!=0.5):
                                matrizs1[i][j]=0
                                matrizs1[j][i]=round(ms1,3)
                            else:
                                matrizs1[i][j]=0.5
                                matrizs1[j][i]=matrizs1[i][j]
                        else:
                            matrizs1[i][j]=0.5
                            matrizs1[j][i]=matrizs1[i][j]
                    else:
                        if(Utils.s1T2(c,b,g, p,q,peso,cri)=="domina") and (Utils.s1T2(e,f,h,p,q,peso,cri)=="domina"):
                            check1 = Utils.discordancias1T2(b,c,g,d,p, q, peso, cri)
                            check2 = Utils.discordancias1T2(f,e,h, d,p, q, peso, cri)
                            if(check1==0.5) or (check2==0.5):
                                matrizs1[i][j]=0.5
                                matrizs1[j][i]=0.5
                            else:
                                matrizs1[i][j]=round(Utils.discordancias1T2(b,c,g, d,p, q, peso, cri),3)
                                matrizs1[j][i]=round(Utils.discordancias1T2(f,e,h, d,p, q, peso, cri),3)
                        elif(Utils.s1T2(c,b,g, p,q,peso,cri)=="domina") and (Utils.s1T2(e,f,h, p,q,peso,cri)!="domina"):
                            ms1 = Utils.discordancias1T2(b,c,g, d,p, q, peso, cri)
                            if(ms1 !=0.5):
                                matrizs1[i][j]=round(ms1,3)
                                matrizs1[j][i]=0
                            else:
                                matrizs1[i][j]=0.5
                                matrizs1[j][i]=0.5
                        elif(Utils.s1T2(c,b,g ,p,q,peso,cri)!="domina") and (Utils.s1T2(e,f,h, p,q,peso,cri)=="domina"):
                            ms1 = Utils.discordancias1T2(f,e,h,d,p, q, peso, cri)
                            if(ms1!=0.5):
                                matrizs1[i][j]=0
                                matrizs1[j][i]=round(ms1,3)
                            else:
                                matrizs1[i][j]=0.5
                                matrizs1[j][i]=matrizs1[i][j]
                        else:
                            matrizs1[i][j]=0.5
                            matrizs1[j][i]=matrizs1[i][j]
                    c=[];b=[];e=[];f=[]
                    if thor.selected_method == 2:
                        g=[];h=[]
        for i in range(alt):
            r1=0.0
            for j in range(alt):
                r1+=matrizs1[i][j]
            rs1.append(round(r1,3))
        for i in range(len(rs1)):
            rs1o.append(rs1[i])
        rs1o.sort()
        rs1o.reverse()
        for i in range (alt):
            for j in range (alt):
                if rs1o[i]==rs1[j]:
                    if alternativas[j] in alternativaso:
                        "nada"
                    else:
                        alternativaso.append(alternativas[j])
        for i in range(alt):
            if(i!=alt-1):
                if rs1o[i]>rs1o[i+1]:
                    alternativaso.insert(2*i+1,">")
                    if contador==0:
                        originals1.append(alternativaso[2*i])
                        originals1.append(">")
                elif rs1o[i]==rs1o[i+1]:
                    alternativaso.insert(2*i+1,"=")
                    if contador==0:
                        originals1.append(alternativaso[2*i])
                        originals1.append("=")
            else:
                if contador==0:
                    originals1.append(alternativaso[2*i])
        if contador==0:
            input_rows = [[alternativas[row]]+["-"]+[str(matrizs1[row][col]) for col in range(alt)] for row in range(alt)]
            bottom = [[alternativas[row]]+["= "]+[str(rs1[row]) for col in range(1)] for row in range(alt)]
            ordem = [[alternativaso[col] for col in range(len(alternativaso))]+[" - Original."] for row in range(1)]
            s1 = Result()
            s1.S_result = [" ".join(input_rows[i]) for i in range(len(input_rows))]
            s1.somatorio = [" ".join(bottom[i]) for i in range(len(bottom))]
            s1.original = " ".join(ordem[0])
            thor.result.append(s1)
        alternativaso=[]
        tca1=0
        ver1=0
        contador+=1
    contador=0;peso3=[]
    for i in range(cri):
        peso[i]=peso4[i]
        peso2[i]=peso4[i]
    #S2
    while contador<1:
        c=[];b=[];e=[];f=[];g=[];h=[];x=0;y=0;w=0;z=0;v=0;t=0;rs2=[];rs2o=[]
        if contador!=0:
            menor=max(peso2)
            for j in range(cri):
                if peso2[j]<menor and peso2.index(menor) not in peso3 and peso2[j]>0:
                    menor=peso2[j]
            peso3.append(peso2.index(menor))
            peso2[peso2.index(menor)]=0
            peso[peso3[contador-1]]=0
        for i in range(alt):
            for j in range(alt):
                if(i<j):
                    for k in range(cri):
                        x=Utils.compara(matriz[i][k],matriz[j][k],p,q,k)
                        y=Utils.dif(matriz[i][k],matriz[j][k])
                        w=Utils.compara(matriz[j][k],matriz[i][k],p,q,k)
                        z=Utils.dif(matriz[j][k],matriz[i][k])
                        v=Utils.ind(pertinencia[k],pertinencia2[i][k],pertinencia2[j][k])
                        t=Utils.ind(pertinencia[k],pertinencia2[j][k],pertinencia2[i][k])
                        b.append(y)
                        c.append(x)
                        e.append(w)
                        f.append(z)
                        g.append(v)
                        h.append(t)
                    if thor.selected_method == 1:
                        if(Utils.s2(c,b,g,peso, cri)=="domina") and (Utils.s2(e,f,h, peso, cri)=="domina"):
                            check1 = Utils.discordancias2(b,c,g, d, peso, cri)
                            check2 = Utils.discordancias2(f,e,h,d,peso, cri)
                            if(check1==0.5) or (check2==0.5):
                                matrizs2[i][j]=0.5
                                matrizs2[j][i]=0.5
                            else:
                                matrizs2[i][j]=round(Utils.discordancias2(b,c,g,d,peso, cri),3)
                                matrizs2[j][i]=round(Utils.discordancias2(f,e,h, d,peso, cri),3)
                        elif(Utils.s2(c,b,g, peso, cri)=="domina") and (Utils.s2(e,f,h, peso, cri)!="domina"):
                            ms2 = Utils.discordancias2(b,c,g, d, peso, cri)
                            if(ms2!=0.5):
                                matrizs2[i][j]=round(ms2,3)
                                matrizs2[j][i]=0
                            else:
                                matrizs2[i][j]=0.5
                                matrizs2[j][i]=0.5
                        elif(Utils.s2(c,b,g, peso, cri)!="domina") and (Utils.s2(e,f,h, peso, cri)=="domina"):
                            ms2 = Utils.discordancias2(f,e,h, d,peso, cri)
                            if(ms2!=0.5):
                                matrizs2[i][j]=0
                                matrizs2[j][i]=round(ms2,3)
                            else:
                                matrizs2[i][j]=0.5
                                matrizs2[j][i]=matrizs2[i][j]
                        else:
                            matrizs2[i][j]=0.5
                            matrizs2[j][i]=matrizs2[i][j]
                    else:
                        if(Utils.s2T2(c,b,g,p,q,peso,cri)=="domina") and (Utils.s2T2(e,f,h,p,q,peso,cri)=="domina"):
                            check1 = Utils.discordancias2T2(b,c,g,d,p, q, peso, cri)
                            check2 = Utils.discordancias2T2(f,e,h,d,p, q, peso, cri)
                            if(check1==0.5) or (check2==0.5):
                                matrizs2[i][j]=0.5
                                matrizs2[j][i]=0.5
                            else:
                                matrizs2[i][j]=round(Utils.discordancias2T2(b,c,g,d,p, q, peso, cri),3)
                                matrizs2[j][i]=round(Utils.discordancias2T2(f,e,h,d,p, q, peso, cri),3)
                        elif(Utils.s2T2(c,b,g,p,q,peso,cri)=="domina") and (Utils.s2T2(e,f,h, p,q,peso,cri)!="domina"):
                            ms2 = Utils.discordancias2T2(b,c,g, d,p,q, peso, cri)
                            if(ms2!=0.5):
                                matrizs2[i][j]=round(ms2,3)
                                matrizs2[j][i]=0
                            else:
                                matrizs2[i][j]=0.5
                                matrizs2[j][i]=0.5
                        elif(Utils.s2T2(c,b,g,p,q,peso,cri)!="domina") and (Utils.s2T2(e,f,h, p,q,peso,cri)=="domina"):
                            ms2 = Utils.discordancias2T2(f,e,h,d,p, q, peso, cri)
                            if(ms2!=0.5):
                                matrizs2[i][j]=0
                                matrizs2[j][i]=round(ms2,3)
                            else:
                                matrizs2[i][j]=0.5
                                matrizs2[j][i]=matrizs2[i][j]
                        else:
                            matrizs2[i][j]=0.5
                            matrizs2[j][i]=matrizs2[i][j]
                    c=[];b=[];e=[];f=[]
                    if thor.selected_method == 2:
                        g=[];h=[]
        for i in range(alt):
            r2=0.0
            for j in range(alt):
                r2+=matrizs2[i][j]
            rs2.append(round(r2,3))
        for i in range(len(rs2)):
            rs2o.append(rs2[i])
        rs2o.sort()
        rs2o.reverse()
        for i in range (alt):
            for j in range (alt):
                if rs2o[i]==rs2[j]:
                    if alternativas[j] in alternativaso:
                        "nada"
                    else:
                        alternativaso.append(alternativas[j])
        for i in range(alt):
            if(i!=alt-1):
                if rs2o[i]>rs2o[i+1]:
                    alternativaso.insert(2*i+1,">")
                    if contador==0:
                        originals2.append(alternativaso[2*i])
                        originals2.append(">")
                elif rs2o[i]==rs2o[i+1]:
                    alternativaso.insert(2*i+1,"=")
                    if contador==0:
                        originals2.append(alternativaso[2*i])
                        originals2.append("=")
            else:
                if contador==0:
                    originals2.append(alternativaso[2*i])
        if contador==0:
            input_rows = [[str(alternativas[row])]+["-"]+[str(matrizs2[row][col]) for col in range(alt)] for row in range(alt)]
            bottom = [[str(alternativas[row])]+["= "]+[str(rs2[row]) for col in range(1)] for row in range(alt)]
            ordem = [[str(alternativaso[col]) for col in range(len(alternativaso))]+[" - Original."] for row in range(1)]
            s2 = Result()
            s2.S_result = [" ".join(input_rows[i]) for i in range(len(input_rows))]
            s2.somatorio = [" ".join(bottom[i]) for i in range(len(bottom))]
            s2.original = " ".join(ordem[0])
            thor.result.append(s2)
        alternativaso=[]
        tca2=0
        ver2=0
        contador+=1
    contador=0;peso3=[]
    for i in range(cri):
        peso[i]=peso4[i]
        peso2[i]=peso4[i]
    #S3
    while contador<1:
        c=[];b=[];e=[];f=[];g=[];h=[];x=0;y=0;w=0;z=0;v=0;t=0;rs3=[];rs3o=[]
        if contador!=0:
            menor=max(peso2)
            for j in range(cri):
                if peso2[j]<menor and peso2.index(menor) not in peso3 and peso2[j]>0:
                    menor=peso2[j]
            peso3.append(peso2.index(menor))
            peso2[peso2.index(menor)]=0
            peso[peso3[contador-1]]=0
        for i in range(alt):
            for j in range(alt):
                if(i<j):
                    for k in range(cri):
                        x=Utils.compara(matriz[i][k],matriz[j][k], p,q,k)
                        y=Utils.dif(matriz[i][k],matriz[j][k])
                        w=Utils.compara(matriz[j][k],matriz[i][k], p,q,k)
                        z=Utils.dif(matriz[j][k],matriz[i][k])
                        v=Utils.ind(pertinencia[k],pertinencia2[i][k],pertinencia2[j][k])
                        t=Utils.ind(pertinencia[k],pertinencia2[j][k],pertinencia2[i][k])
                        b.append(y)
                        c.append(x)
                        e.append(w)
                        f.append(z)
                        g.append(v)
                        h.append(t)
                    if thor.selected_method == 1:
                        if(Utils.s3(c,b,g, peso,cri)=="domina") and (Utils.s3(e,f,h, peso, cri)=="domina"):
                            check1 = Utils.discordancias3(b,c,g,d,peso, cri)
                            check2 = Utils.discordancias3(f,e,h,d,peso, cri)
                            if(check1==0.5) or (check2==0.5):
                                matrizs3[i][j]=0.5
                                matrizs3[j][i]=0.5
                            else:
                                matrizs3[i][j]=round(Utils.discordancias3(b,c,g,d,peso, cri),3)
                                matrizs3[j][i]=round(Utils.discordancias3(f,e,h, d,peso, cri),3)
                        elif(Utils.s3(c,b,g, peso, cri)=="domina") and (Utils.s3(e,f,h, peso, cri)!="domina"):
                            ms3 = Utils.discordancias3(b,c,g,d,peso, cri)
                            if(ms3!=0.5):
                                matrizs3[i][j]=round(ms3,3)
                                matrizs3[j][i]=0
                            else:
                                matrizs3[i][j]=0.5
                                matrizs3[j][i]=0.5
                        elif(Utils.s3(c,b,g,peso,cri)!="domina") and (Utils.s3(e,f,h, peso, cri)=="domina"):
                            ms3 = Utils.discordancias3(f,e,h, d,peso, cri)
                            if(ms3!=0.5):
                                matrizs3[i][j]=0
                                matrizs3[j][i]=round(ms3,3)
                            else:
                                matrizs3[i][j]=0.5
                                matrizs3[j][i]=matrizs3[i][j]
                        else:
                            matrizs3[i][j]=0.5
                            matrizs3[j][i]=matrizs3[i][j]
                    else:
                        if(Utils.s3T2(c,b,g,p,q,peso,cri)=="domina") and (Utils.s3T2(e,f,h,p,q,peso,cri)=="domina"):
                            check1 = Utils.discordancias3T2(b,c,g,d,p, q, peso, cri)
                            check2 = Utils.discordancias3T2(f,e,h,d,p, q, peso, cri)
                            if(check1==0.5) or (check2==0.5):
                                matrizs3[i][j]=0.5
                                matrizs3[j][i]=0.5
                            else:
                                matrizs3[i][j]=round(Utils.discordancias3T2(b,c,g,d,p, q, peso, cri),3)
                                matrizs3[j][i]=round(Utils.discordancias3T2(f,e,h,d,p, q, peso, cri),3)
                        elif(Utils.s3T2(c,b,g,p,q,peso,cri)=="domina") and (Utils.s3T2(e,f,h,p,q,peso,cri)!="domina"):
                            ms3 = Utils.discordancias3T2(b,c,g,d,p, q, peso, cri)
                            if(ms3!=0.5):
                                matrizs3[i][j]=round(ms3,3)
                                matrizs3[j][i]=0
                            else:
                                matrizs3[i][j]=0.5
                                matrizs3[j][i]=0.5
                        elif(Utils.s3T2(c,b,g,p,q,peso,cri)!="domina") and (Utils.s3T2(e,f,h,p,q,peso,cri)=="domina"):
                            ms3 = Utils.discordancias3T2(f,e,h,d,p, q, peso, cri)
                            if(ms3!=0.5):
                                matrizs3[i][j]=0
                                matrizs3[j][i]=round(ms3,3)
                            else:
                                matrizs3[i][j]=0.5
                                matrizs3[j][i]=matrizs3[i][j]
                        else:
                            matrizs3[i][j]=0.5
                            matrizs3[j][i]=matrizs3[i][j]
                    c=[];b=[];e=[];f=[]
                    if thor.selected_method == 2:
                        g=[];h=[]
        for i in range(alt):
            r3=0.0
            for j in range(alt):
                r3+=matrizs3[i][j]
            rs3.append(round(r3,3))
        for i in range(len(rs3)):
            rs3o.append(rs3[i])
        rs3o.sort()
        rs3o.reverse()
        for i in range (alt):
            for j in range (alt):
                if rs3o[i]==rs3[j]:
                    if alternativas[j] in alternativaso:
                        "nada"
                    else:
                        alternativaso.append(alternativas[j])
        for i in range(alt):
            if(i!=alt-1):
                if rs3o[i]>rs3o[i+1]:
                    alternativaso.insert(2*i+1,">")
                    if contador==0:
                        originals3.append(alternativaso[2*i])
                        originals3.append(">")
                elif rs3o[i]==rs3o[i+1]:
                    alternativaso.insert(2*i+1,"=")
                    if contador==0:
                        originals3.append(alternativaso[2*i])
                        originals3.append("=")
            else:
                if contador==0:
                    originals3.append(alternativaso[2*i])
        if contador==0:
            input_rows = [[str(alternativas[row])]+["-"]+[str(matrizs3[row][col]) for col in range(alt)] for row in range(alt)]
            bottom = [[str(alternativas[row])]+["= "]+[str(rs3[row]) for col in range(1)] for row in range(alt)]
            ordem = [[str(alternativaso[col]) for col in range(len(alternativaso))]+[" - Original."] for row in range(1)]
            s3 = Result()
            s3.S_result = [" ".join(input_rows[i]) for i in range(len(input_rows))]
            s3.somatorio = [" ".join(bottom[i]) for i in range(len(bottom))]
            s3.original = " ".join(ordem[0])
            thor.result.append(s3)
            alternativaso=[]
        tca3=0
        ver3=0
        contador+=1
    #TCA S1
    if usartca!=1:
        contador=1;peso3=[]
        c=[];b=[];e=[];f=[];g=[];h=[];menor=max(peso)
        while contador<cri+1:
            c=[];b=[];e=[];f=[];g=[];h=[];x=0;y=0;w=0;z=0;v=0;t=0;rs1=[];rs1o=[];
            if contador!=0 and usartca!=1:
                menor=max(peso2)
                for j in range(cri):
                    if peso2[j]<menor and peso2.index(menor) not in peso3 and peso2[j]>0:
                        menor=peso2[j]
                peso3.append(peso2.index(menor))
                peso2[peso2.index(menor)]=0
                peso[peso3[contador-1]]=0
            for i in range(alt):
                for j in range(alt):
                    if(i<j):
                        for k in range(cri):
                            x=Utils.compara(matriz[i][k],matriz[j][k], p,q , k)
                            y=Utils.dif(matriz[i][k],matriz[j][k])
                            w=Utils.compara(matriz[j][k],matriz[i][k], p, q, k)
                            z=Utils.dif(matriz[j][k],matriz[i][k])
                            v=Utils.ind(pertinencia[k],pertinencia2[i][k],pertinencia2[j][k])
                            t=Utils.ind(pertinencia[k],pertinencia2[j][k],pertinencia2[i][k])
                            b.append(y)
                            c.append(x)
                            e.append(w)
                            f.append(z)
                            g.append(v)
                            h.append(t)
                        if thor.selected_method == 1:
                            if(Utils.s1(c,b,g, peso, cri)=="domina") and (Utils.s1(e,f,h, peso, cri)=="domina"):
                                check1 = Utils.discordancias1(b,c,g,d,peso, cri)
                                check2 = Utils.discordancias1(f,e,h,d,peso, cri)
                                if(check1==0.5) or (check2==0.5):
                                    matrizs1[i][j]=0.5
                                    matrizs1[j][i]=0.5
                                else:
                                    matrizs1[i][j]=round(Utils.discordancias1(b,c,g,d,peso, cri),3)
                                    matrizs1[j][i]=round(Utils.discordancias1(f,e,h,d,peso, cri),3)
                            elif(Utils.s1(c,b,g, peso, cri)=="domina") and (Utils.s1(e,f,h, peso, cri)!="domina"):
                                ms1 = Utils.discordancias1(b,c,g,d,peso, cri)
                                if(ms1!=0.5):
                                    matrizs1[i][j]=round(ms1,3)
                                    matrizs1[j][i]=0
                                else:
                                    matrizs1[i][j]=0.5
                                    matrizs1[j][i]=0.5
                            elif(Utils.s1(c,b,g,peso, cri)!="domina") and (Utils.s1(e,f,h, peso, cri)=="domina"):
                                ms1 = Utils.discordancias1(f,e,h,d,peso, cri)
                                if(ms1!=0.5):
                                    matrizs1[i][j]=0
                                    matrizs1[j][i]=round(ms1,3)
                                else:
                                    matrizs1[i][j]=0.5
                                    matrizs1[j][i]=matrizs1[i][j]
                            else:
                                matrizs1[i][j]=0.5
                                matrizs1[j][i]=matrizs1[i][j]
                        else:
                            if(Utils.s1T2(c,b,g, p,q,peso,cri)=="domina") and (Utils.s1T2(e,f,h,p,q,peso,cri)=="domina"):
                                check1 = Utils.discordancias1T2(b,c,g,d,p, q, peso, cri)
                                check2 = Utils.discordancias1T2(f,e,h, d,p, q, peso, cri)
                                if(check1==0.5) or (check2==0.5):
                                    matrizs1[i][j]=0.5
                                    matrizs1[j][i]=0.5
                                else:
                                    matrizs1[i][j]=round(Utils.discordancias1T2(b,c,g, d,p, q, peso, cri),3)
                                    matrizs1[j][i]=round(Utils.discordancias1T2(f,e,h, d,p, q, peso, cri),3)
                            elif(Utils.s1T2(c,b,g, p,q,peso,cri)=="domina") and (Utils.s1T2(e,f,h, p,q,peso,cri)!="domina"):
                                ms1 = Utils.discordancias1T2(b,c,g, d,p, q, peso, cri)
                                if(ms1 !=0.5):
                                    matrizs1[i][j]=round(ms1,3)
                                    matrizs1[j][i]=0
                                else:
                                    matrizs1[i][j]=0.5
                                    matrizs1[j][i]=0.5
                            elif(Utils.s1T2(c,b,g ,p,q,peso,cri)!="domina") and (Utils.s1T2(e,f,h, p,q,peso,cri)=="domina"):
                                ms1 = Utils.discordancias1T2(f,e,h,d,p, q, peso, cri)
                                if(ms1!=0.5):
                                    matrizs1[i][j]=0
                                    matrizs1[j][i]=round(ms1,3)
                                else:
                                    matrizs1[i][j]=0.5
                                    matrizs1[j][i]=matrizs1[i][j]
                            else:
                                matrizs1[i][j]=0.5
                                matrizs1[j][i]=matrizs1[i][j]
                        c=[];b=[];e=[];f=[]
            for i in range(alt):
                r1=0.0
                for j in range(alt):
                    r1+=matrizs1[i][j]
                rs1.append(round(r1,3))
            for i in range(len(rs1)):
                rs1o.append(rs1[i])
            rs1o.sort()
            rs1o.reverse()
            for i in range (alt):
                for j in range (alt):
                    if rs1o[i]==rs1[j]:
                        if alternativas[j] in alternativaso:
                            "nada"
                        else:
                            alternativaso.append(alternativas[j])
            for i in range(alt):
                if(i!=alt-1):
                    if rs1o[i]>rs1o[i+1]:
                        alternativaso.insert(2*i+1,">")
                        if contador==0:
                            originals1.append(alternativaso[2*i])
                            originals1.append(">")
                    elif rs1o[i]==rs1o[i+1]:
                        alternativaso.insert(2*i+1,"=")
                        if contador==0:
                            originals1.append(alternativaso[2*i])
                            originals1.append("=")
                else:
                    if contador==0:
                        originals1.append(alternativaso[2*i])
            if contador==0:
                input_rows = [[str(alternativas[row])]+[str("-")]+[str(matrizs1[row][col]) for col in range(alt)] for row in range(alt)]
                bottom = [[str(alternativas[row])]+[str("= ")]+[str(rs1[row]) for col in range(1)] for row in range(alt)]
                ordem = [[str(alternativaso[col]) for col in range(len(alternativaso))]+[str(" - Original.")] for row in range(1)]
                layout = input_rows + bottom + ordem
                #window = sg.Window('Resultados de S1', layout, font='Courier 12')
            elif contador!=0 and usartca!=1:
                result = ResultTca()
                result.sub_title = f'Criteria analyzed {criterios[peso3[contador-1]]}'
                result.title = 'S1 result'
                input_rows = [[str(alternativas[row])]+[str("-")]+[str(matrizs1[row][col]) for col in range(alt)] for row in range(alt)]
                bottom = [[str(alternativas[row])]+[str("= ")]+[str(rs1[row]) for col in range(1)] for row in range(alt)]
                ordem = [[str(alternativaso[col]) for col in range(len(alternativaso))]+[str(" - Without the criteria.")] for row in range(1)]
                ordem2 = [[str(originals1[col]) for col in range(len(originals1))]+[str(" - Original.")] for row in range(1)]
                result.S_result = [" ".join(input_rows[i]) for i in range(len(input_rows))]
                result.somatorio = [" ".join(bottom[i]) for i in range(len(bottom))]
                result.original = " ".join(ordem[0])
                result.original2 = " ".join(ordem2[0])
                thor.result_tca_s1.append(result)
            if contador!=0 and usartca!=1:
                for i in range(len(alternativaso)):
                    if alternativaso[i]==originals1[i]:
                        tca1+=1
                if (tca1==len(alternativaso)):
                    cris1.append(criterios[peso3[contador-1]])
                    if (criterios[peso3[contador-1]]) not in cristotal:
                        cristotal.append(criterios[peso3[contador-1]])
                    ver1=1
            if contador!=0 and usartca!=1:
                if ver1!=1:
                    peso[peso3[contador-1]]=peso4[peso3[contador-1]]
            alternativaso=[]
            tca1=0
            ver1=0
            contador+=1
        if usartca!=1: #3
            if len(cris1)==0:
                thor.tca_s1_citerio_removed = "No criteria can be removed"
            else:
                ordem = [[str(cris1[row]) for col in range(1)] for row in range(len(cris1))]
                c = [" - ".join(ordem[i]) for i in range(len(ordem))]
                c = " - ".join(c)
                thor.tca_s1_citerio_removed = "Criteria can be removed : " + c
        contador=1;peso3=[]
        for i in range(cri):
            peso[i]=peso4[i]
            peso2[i]=peso4[i]
    #S2 TCA
    if usartca!=1:
        while contador<cri+1:
            c=[];b=[];e=[];f=[];g=[];h=[];x=0;y=0;w=0;z=0;v=0;t=0;rs2=[];rs2o=[]
            if contador!=0 and usartca!=1:
                menor=max(peso2)
                for j in range(cri):
                    if peso2[j]<menor and peso2.index(menor) not in peso3 and peso2[j]>0:
                        menor=peso2[j]
                peso3.append(peso2.index(menor))
                peso2[peso2.index(menor)]=0
                peso[peso3[contador-1]]=0
            for i in range(alt):
                for j in range(alt):
                    if(i<j):
                        for k in range(cri):
                            x=Utils.compara(matriz[i][k],matriz[j][k],p,q,k)
                            y=Utils.dif(matriz[i][k],matriz[j][k])
                            w=Utils.compara(matriz[j][k],matriz[i][k],p,q,k)
                            z=Utils.dif(matriz[j][k],matriz[i][k])
                            v=Utils.ind(pertinencia[k],pertinencia2[i][k],pertinencia2[j][k])
                            t=Utils.ind(pertinencia[k],pertinencia2[j][k],pertinencia2[i][k])
                            b.append(y)
                            c.append(x)
                            e.append(w)
                            f.append(z)
                            g.append(v)
                            h.append(t)
                        if thor.selected_method == 1:
                            if(Utils.s2(c,b,g,peso, cri)=="domina") and (Utils.s2(e,f,h, peso, cri)=="domina"):
                                check1 = Utils.discordancias2(b,c,g, d, peso, cri)
                                check2 = Utils.discordancias2(f,e,h,d,peso, cri)
                                if(check1==0.5) or (check2==0.5):
                                    matrizs2[i][j]=0.5
                                    matrizs2[j][i]=0.5
                                else:
                                    matrizs2[i][j]=round(Utils.discordancias2(b,c,g,d,peso, cri),3)
                                    matrizs2[j][i]=round(Utils.discordancias2(f,e,h, d,peso, cri),3)
                            elif(Utils.s2(c,b,g, peso, cri)=="domina") and (Utils.s2(e,f,h, peso, cri)!="domina"):
                                ms2 = Utils.discordancias2(b,c,g, d, peso, cri)
                                if(ms2!=0.5):
                                    matrizs2[i][j]=round(ms2,3)
                                    matrizs2[j][i]=0
                                else:
                                    matrizs2[i][j]=0.5
                                    matrizs2[j][i]=0.5
                            elif(Utils.s2(c,b,g, peso, cri)!="domina") and (Utils.s2(e,f,h, peso, cri)=="domina"):
                                ms2 = Utils.discordancias2(f,e,h, d,peso, cri)
                                if(ms2!=0.5):
                                    matrizs2[i][j]=0
                                    matrizs2[j][i]=round(ms2,3)
                                else:
                                    matrizs2[i][j]=0.5
                                    matrizs2[j][i]=matrizs2[i][j]
                            else:
                                matrizs2[i][j]=0.5
                                matrizs2[j][i]=matrizs2[i][j]
                        else:
                            if(Utils.s2T2(c,b,g,p,q,peso,cri)=="domina") and (Utils.s2T2(e,f,h,p,q,peso,cri)=="domina"):
                                check1 = Utils.discordancias2T2(b,c,g,d,p, q, peso, cri)
                                check2 = Utils.discordancias2T2(f,e,h,d,p, q, peso, cri)
                                if(check1==0.5) or (check2==0.5):
                                    matrizs2[i][j]=0.5
                                    matrizs2[j][i]=0.5
                                else:
                                    matrizs2[i][j]=round(Utils.discordancias2T2(b,c,g,d,p, q, peso, cri),3)
                                    matrizs2[j][i]=round(Utils.discordancias2T2(f,e,h,d,p, q, peso, cri),3)
                            elif(Utils.s2T2(c,b,g,p,q,peso,cri)=="domina") and (Utils.s2T2(e,f,h, p,q,peso,cri)!="domina"):
                                ms2 = Utils.discordancias2T2(b,c,g, d,p,q, peso, cri)
                                if(ms2!=0.5):
                                    matrizs2[i][j]=round(ms2,3)
                                    matrizs2[j][i]=0
                                else:
                                    matrizs2[i][j]=0.5
                                    matrizs2[j][i]=0.5
                            elif(Utils.s2T2(c,b,g,p,q,peso,cri)!="domina") and (Utils.s2T2(e,f,h, p,q,peso,cri)=="domina"):
                                ms2 = Utils.discordancias2T2(f,e,h,d,p, q, peso, cri)
                                if(ms2!=0.5):
                                    matrizs2[i][j]=0
                                    matrizs2[j][i]=round(ms2,3)
                                else:
                                    matrizs2[i][j]=0.5
                                    matrizs2[j][i]=matrizs2[i][j]
                            else:
                                matrizs2[i][j]=0.5
                                matrizs2[j][i]=matrizs2[i][j]
                        c=[];b=[];e=[];f=[]
            for i in range(alt):
                r2=0.0
                for j in range(alt):
                    r2+=matrizs2[i][j]
                rs2.append(round(r2,3))
            for i in range(len(rs2)):
                rs2o.append(rs2[i])
            rs2o.sort()
            rs2o.reverse()
            for i in range (alt):
                for j in range (alt):
                    if rs2o[i]==rs2[j]:
                        if alternativas[j] in alternativaso:
                            "nada"
                        else:
                            alternativaso.append(alternativas[j])
            for i in range(alt):
                if(i!=alt-1):
                    if rs2o[i]>rs2o[i+1]:
                        alternativaso.insert(2*i+1,">")
                        if contador==0:
                            originals2.append(alternativaso[2*i])
                            originals2.append(">")
                    elif rs2o[i]==rs2o[i+1]:
                        alternativaso.insert(2*i+1,"=")
                        if contador==0:
                            originals2.append(alternativaso[2*i])
                            originals2.append("=")
                else:
                    if contador==0:
                        originals2.append(alternativaso[2*i])
            if contador==0:
                input_rows = [[str(alternativas[row])]+[str("-")]+[str(matrizs2[row][col]) for col in range(alt)] for row in range(alt)]
                bottom = [[str(alternativas[row])]+[str("= ")]+[str(rs2[row]) for col in range(1)] for row in range(alt)]
                ordem = [[str(alternativaso[col]) for col in range(len(alternativaso))]+[str(" - Original.")] for row in range(1)]
                layout = input_rows + bottom + ordem
                #window = sg.Window('Resultados de S2', layout, font='Courier 12')
            elif contador!=0 and usartca!=1:
                result = ResultTca()
                result.sub_title = f'Criteria analyzed {criterios[peso3[contador-1]]}'
                result.title = 'S2 result'
                input_rows = [[str(alternativas[row])]+[str("-")]+[str(matrizs2[row][col]) for col in range(alt)] for row in range(alt)]
                bottom = [[str(alternativas[row])]+[str("= ")]+[str(rs2[row]) for col in range(1)] for row in range(alt)]
                ordem = [[str(alternativaso[col]) for col in range(len(alternativaso))]+[str(" - Without the criteria.")] for row in range(1)]
                ordem2 = [[str(originals2[col]) for col in range(len(originals2))]+[str(" - Original.")] for row in range(1)]
                result.S_result = [" ".join(input_rows[i]) for i in range(len(input_rows))]
                result.somatorio = [" ".join(bottom[i]) for i in range(len(bottom))]
                result.original = " ".join(ordem[0])
                result.original2 = " ".join(ordem2[0])
                thor.result_tca_s2.append(result)
            if contador!=0 and usartca!=1:
                for i in range(len(alternativaso)):
                    if alternativaso[i]==originals2[i]:
                        tca2+=1
            if (tca2==len(alternativaso)):
                cris2.append(criterios[peso3[contador-1]])
                if (criterios[peso3[contador-1]]) not in cristotal:
                    cristotal.append(criterios[peso3[contador-1]])
                ver2=1
            if contador!=0 and usartca!=1:
                if ver2!=1:
                    peso[peso3[contador-1]]=peso4[peso3[contador-1]]
            alternativaso=[]
            tca2=0
            ver2=0
            contador+=1
        if usartca!=1:
            if len(cris2)==0:
               thor.tca_s2_citerio_removed = "No criteria can be removed"
            else:
                ordem = [[str(cris2[row]) for col in range(1)] for row in range(len(cris2))]
                c = [" - ".join(ordem[i]) for i in range(len(ordem))]
                thor.tca_s2_citerio_removed = "Criteria can be removed :" + " - ".join(c)
        contador=1;peso3=[]
        for i in range(cri):
            peso[i]=peso4[i]
            peso2[i]=peso4[i]
    #S3
    if usartca!=1:
        while contador<cri+1:
            c=[];b=[];e=[];f=[];g=[];h=[];x=0;y=0;w=0;z=0;v=0;t=0;rs3=[];rs3o=[]
            if contador!=0 and usartca!=1:
                menor=max(peso2)
                for j in range(cri):
                    if peso2[j]<menor and peso2.index(menor) not in peso3 and peso2[j]>0:
                        menor=peso2[j]
                peso3.append(peso2.index(menor))
                peso2[peso2.index(menor)]=0
                peso[peso3[contador-1]]=0
            for i in range(alt):
                for j in range(alt):
                    if(i<j):
                        for k in range(cri):
                            x=Utils.compara(matriz[i][k],matriz[j][k],p,q,k)
                            y=Utils.dif(matriz[i][k],matriz[j][k])
                            w=Utils.compara(matriz[j][k],matriz[i][k],p,q,k)
                            z=Utils.dif(matriz[j][k],matriz[i][k])
                            v=Utils.ind(pertinencia[k],pertinencia2[i][k],pertinencia2[j][k])
                            t=Utils.ind(pertinencia[k],pertinencia2[j][k],pertinencia2[i][k])
                            b.append(y)
                            c.append(x)
                            e.append(w)
                            f.append(z)
                            g.append(v)
                            h.append(t)
                        if thor.selected_method == 1:
                            if(Utils.s3(c,b,g, peso,cri)=="domina") and (Utils.s3(e,f,h, peso, cri)=="domina"):
                                check1 = Utils.discordancias3(b,c,g,d,peso, cri)
                                check2 = Utils.discordancias3(f,e,h,d,peso, cri)
                                if(check1==0.5) or (check2==0.5):
                                    matrizs3[i][j]=0.5
                                    matrizs3[j][i]=0.5
                                else:
                                    matrizs3[i][j]=round(Utils.discordancias3(b,c,g,d,peso, cri),3)
                                    matrizs3[j][i]=round(Utils.discordancias3(f,e,h, d,peso, cri),3)
                            elif(Utils.s3(c,b,g, peso, cri)=="domina") and (Utils.s3(e,f,h, peso, cri)!="domina"):
                                ms3 = Utils.discordancias3(b,c,g,d,peso, cri)
                                if(ms3!=0.5):
                                    matrizs3[i][j]=round(ms3,3)
                                    matrizs3[j][i]=0
                                else:
                                    matrizs3[i][j]=0.5
                                    matrizs3[j][i]=0.5
                            elif(Utils.s3(c,b,g,peso,cri)!="domina") and (Utils.s3(e,f,h, peso, cri)=="domina"):
                                ms3 = Utils.discordancias3(f,e,h, d,peso, cri)
                                if(ms3!=0.5):
                                    matrizs3[i][j]=0
                                    matrizs3[j][i]=round(ms3,3)
                                else:
                                    matrizs3[i][j]=0.5
                                    matrizs3[j][i]=matrizs3[i][j]
                            else:
                                matrizs3[i][j]=0.5
                                matrizs3[j][i]=matrizs3[i][j]
                        else:
                            if(Utils.s3T2(c,b,g,p,q,peso,cri)=="domina") and (Utils.s3T2(e,f,h,p,q,peso,cri)=="domina"):
                                check1 = Utils.discordancias3T2(b,c,g,d,p, q, peso, cri)
                                check2 = Utils.discordancias3T2(f,e,h,d,p, q, peso, cri)
                                if(check1==0.5) or (check2==0.5):
                                    matrizs3[i][j]=0.5
                                    matrizs3[j][i]=0.5
                                else:
                                    matrizs3[i][j]=round(Utils.discordancias3T2(b,c,g,d,p, q, peso, cri),3)
                                    matrizs3[j][i]=round(Utils.discordancias3T2(f,e,h,d,p, q, peso, cri),3)
                            elif(Utils.s3T2(c,b,g,p,q,peso,cri)=="domina") and (Utils.s3T2(e,f,h,p,q,peso,cri)!="domina"):
                                ms3 = Utils.discordancias3T2(b,c,g,d,p, q, peso, cri)
                                if(ms3!=0.5):
                                    matrizs3[i][j]=round(ms3,3)
                                    matrizs3[j][i]=0
                                else:
                                    matrizs3[i][j]=0.5
                                    matrizs3[j][i]=0.5
                            elif(Utils.s3T2(c,b,g,p,q,peso,cri)!="domina") and (Utils.s3T2(e,f,h,p,q,peso,cri)=="domina"):
                                ms3 = Utils.discordancias3T2(f,e,h,d,p, q, peso, cri)
                                if(ms3!=0.5):
                                    matrizs3[i][j]=0
                                    matrizs3[j][i]=round(ms3,3)
                                else:
                                    matrizs3[i][j]=0.5
                                    matrizs3[j][i]=matrizs3[i][j]
                            else:
                                matrizs3[i][j]=0.5
                                matrizs3[j][i]=matrizs3[i][j]
                        c=[];b=[];e=[];f=[]
            for i in range(alt):
                r3=0.0
                for j in range(alt):
                    r3+=matrizs3[i][j]
                rs3.append(round(r3,3))
            for i in range(len(rs3)):
                rs3o.append(rs3[i])
            rs3o.sort()
            rs3o.reverse()
            for i in range (alt):
                for j in range (alt):
                    if rs3o[i]==rs3[j]:
                        if alternativas[j] in alternativaso:
                            "nada"
                        else:
                            alternativaso.append(alternativas[j])
            for i in range(alt):
                if(i!=alt-1):
                    if rs3o[i]>rs3o[i+1]:
                        alternativaso.insert(2*i+1,">")
                        if contador==0:
                            originals3.append(alternativaso[2*i])
                            originals3.append(">")
                    elif rs3o[i]==rs3o[i+1]:
                        alternativaso.insert(2*i+1,"=")
                        if contador==0:
                            originals3.append(alternativaso[2*i])
                            originals3.append("=")
                else:
                    if contador==0:
                        originals3.append(alternativaso[2*i])
            if contador==0:
                input_rows = [[str(alternativas[row])]+[str("-")]+[str(matrizs3[row][col]) for col in range(alt)] for row in range(alt)]
                bottom = [[str(alternativas[row])]+[str("= ")]+[str(rs3[row]) for col in range(1)] for row in range(alt)]
                ordem = [[str(alternativaso[col]) for col in range(len(alternativaso))]+[str(" - Original.")] for row in range(1)]
            elif contador!=0 and usartca!=1:
                result = ResultTca()
                result.sub_title = f'Criteria analyzed {criterios[peso3[contador-1]]}'
                result.title = 'S3 result'
                input_rows = [[str(alternativas[row])]+[str("-")]+[str(matrizs3[row][col]) for col in range(alt)] for row in range(alt)]
                bottom = [[str(alternativas[row])]+[str("= ")]+[str(rs3[row]) for col in range(1)] for row in range(alt)]
                ordem = [[str(alternativaso[col]) for col in range(len(alternativaso))]+[str(" - Without the criteria.")] for row in range(1)]
                ordem2 = [[str(originals3[col]) for col in range(len(originals3))]+[str(" - Original.")] for row in range(1)]
                result.S_result = [" ".join(input_rows[i]) for i in range(len(input_rows))]
                result.somatorio = [" ".join(bottom[i]) for i in range(len(bottom))]
                result.original = " ".join(ordem[0])
                result.original2 = " ".join(ordem2[0])
                thor.result_tca_s3.append(result)
            if contador!=0 and usartca!=1:
                for i in range(len(alternativaso)):
                    if alternativaso[i]==originals3[i]:
                        tca3+=1
                if (tca3==len(alternativaso)):
                    cris3.append(criterios[peso3[contador-1]])
                    if (criterios[peso3[contador-1]]) not in cristotal:
                        cristotal.append(criterios[peso3[contador-1]])
                    ver3=1
            if contador!=0 and usartca!=1:
                if ver3!=1:
                    peso[peso3[contador-1]]=peso4[peso3[contador-1]]
            alternativaso=[]
            tca3=0
            ver3=0
            contador+=1
        if usartca!=1:
            if len(cris3)==0:
                thor.tca_s3_citerio_removed = "No criteria can be removed"
            else:
                ordem = [[str(cris3[row]) for col in range(1)] for row in range(len(cris3))]
                c = [" - ".join(ordem[i]) for i in range(len(ordem))]
                thor.tca_s3_citerio_removed = "Criteria can be removed :" + " - ".join(c)

    for r in range(len(thor.result)):
        title = f"Result S{r+1}"
        pdf.cell(0, 5, title, 0, 1)
        obj = thor.result[r]
        for i in range(len(obj.S_result)):
            pdf.cell(0, 5, f"{' '.join(str(x) for x in obj.S_result[i])}", 0, 1)
        pdf.cell(0, 5, '', 0, 1)
        for i in range(len(obj.somatorio)):
            pdf.cell(0, 5, f"{' '.join(str(x) for x in obj.somatorio[i])}", 0, 1)
        pdf.cell(0, 5, obj.original, 0, 1)
        pdf.cell(0, 5, '', 0, 1)

    if usartca!=1:
        pdf.cell(0, 5, "TCA Result section:", 0, 1)
        for r in range(len(thor.result_tca_s1)):
            obj = thor.result_tca_s1[r]
            pdf.cell(0, 5, obj.title, 0, 1)
            pdf.cell(0, 5, obj.sub_title, 0, 1)
            for i in range(len(obj.S_result)):
                pdf.cell(0, 5, f"{' '.join(str(x) for x in obj.S_result[i])}", 0, 1)
            pdf.cell(0, 5, '', 0, 1)
            for i in range(len(obj.somatorio)):
                pdf.cell(0, 5, f"{' '.join(str(x) for x in obj.somatorio[i])}", 0, 1)
            pdf.cell(0, 5, obj.original, 0, 1)
            if obj.original2:
                pdf.cell(0, 5, obj.original2, 0, 1)
            pdf.cell(0, 5, '', 0, 1)

        for r in range(len(thor.result_tca_s2)):
            obj = thor.result_tca_s2[r]
            pdf.cell(0, 5, obj.title, 0, 1)
            pdf.cell(0, 5, obj.sub_title, 0, 1)
            for i in range(len(obj.S_result)):
                pdf.cell(0, 5, f"{' '.join(str(x) for x in obj.S_result[i])}", 0, 1)
            pdf.cell(0, 5, '', 0, 1)
            for i in range(len(obj.somatorio)):
                pdf.cell(0, 5, f"{' '.join(str(x) for x in obj.somatorio[i])}", 0, 1)
            pdf.cell(0, 5, obj.original, 0, 1)
            if obj.original2:
                pdf.cell(0, 5, obj.original2, 0, 1)
            pdf.cell(0, 5, '', 0, 1)

        for r in range(len(thor.result_tca_s3)):
            obj = thor.result_tca_s3[r]
            pdf.cell(0, 5, obj.title, 0, 1)
            pdf.cell(0, 5, obj.sub_title, 0, 1)
            for i in range(len(obj.S_result)):
                pdf.cell(0, 5, f"{' '.join(str(x) for x in obj.S_result[i])}", 0, 1)
            pdf.cell(0, 5, '', 0, 1)
            for i in range(len(obj.somatorio)):
                pdf.cell(0, 5, f"{' '.join(str(x) for x in obj.somatorio[i])}", 0, 1)
            pdf.cell(0, 5, obj.original, 0, 1)
            if obj.original2:
                pdf.cell(0, 5, obj.original2, 0, 1)
            pdf.cell(0, 5, '', 0, 1)
    if thor.usartca and thor.user_pertinence:
        tcan = TcaNebulosa()
        pdf.cell(0, 5, "Nebulosa TCA original result section:", 0, 1)
        head = [["Weights - "]+[str(pertinencia[col]) for col in range(cri)]]
        input_rows = [[" "]+[str(alternativas[row])]+["- "]+[str(pertinencia2[row][col]) for col in range(cri)] for row in range(alt)]
        medias = [["Average of weights :"]+[str(medtcan[0])]]
        mediaalt = [["Average of "]+[str(alternativas[row])]+["-"]+[str(medtcan[row+1]) for col in range(1)] for row in range(alt)]
        mediamedias = [["Average of weights with"]+[str(alternativas[row])]+["-"]+[str(medtcan[row+1+alt])for col in range(1)] for row in range(alt)]
        tcan.head = [" ".join(head[i]) for i in range(len(head))]
        tcan.input_rows = [" ".join(input_rows[i]) for i in range(len(input_rows))]
        tcan.medias = [" ".join(medias[i]) for i in range(len(medias))]
        tcan.mediamedias = [" ".join(mediamedias[i]) for i in range(len(mediamedias))]
        tcan.mediaalt = [" ".join(mediaalt[i]) for i in range(len(mediaalt))]

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

        for i in range(len(cristotal)):
            neb=0;rtcan=[];rt2=[]
            pos=criterios.index(cristotal[i])
            pertinencia.pop(pos)
            z=round(Utils.media(pertinencia),4)
            rtcan.append(z)
            for k in range(alt):
                pertinencia2[k].pop(pos)
                z=round(Utils.media(pertinencia2[k]),4)
                rtcan.append(z)
                z=round(((rtcan[k+1]+rtcan[0])/2),4)
                rt2.append(z)
            tcanCalc = TcaNebulosa()
            pdf.cell(0, 5, "Nebulosa TCA calculated result section:", 0, 1)
            header =  [['Removing criteria']+[str(cristotal[i])]]
            head = [["Weights - "]+[str(pertinencia[col]) for col in range(len(pertinencia))]]
            input_rows = [[" "]+[str(alternativas[row])]+["- "]+[str(pertinencia2[row][col]) for col in range(len(pertinencia))] for row in range(alt)]
            medias = [["Average of weights :"]+[str(rtcan[0])]]
            mediaalt = [["Average of "]+[str(alternativas[row])]+["-"]+[str(rtcan[row+1]) for col in range(1)] for row in range(alt)]
            mediamedias = [["Average of weights with"]+[str(alternativas[row])]+["-"]+[str(rt2[row])for col in range(1)] for row in range(alt)]

            tcanCalc.title =[" ".join(header[i]) for i in range(len(header))]
            tcanCalc.head = [" ".join(head[i]) for i in range(len(head))]
            tcanCalc.input_rows = [" ".join(input_rows[i]) for i in range(len(input_rows))]
            tcanCalc.medias = [" ".join(medias[i]) for i in range(len(medias))]
            tcanCalc.mediamedias = [" ".join(mediamedias[i]) for i in range(len(mediamedias))]
            tcanCalc.mediaalt = [" ".join(mediaalt[i]) for i in range(len(mediaalt))]
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
                pertinencia2[k].insert(pos,pertinencia2tca[k][pos])
            pertinencia.insert(pos,pertinenciatca[pos])
            for j in range(len(rt2)):
                rtcan.append(rt2[j])
            for j in range(len(rtcan)):
                if rtcan[j]>medtcan[j]:
                    neb+=1
            if neb==len(rtcan):
                tcaneb.append(cristotal[i])
        if len(tcaneb)==0:
            thor.removedTcaN = "By TCA nebulosa no criteria can be removed"
        else:
            head = [[str(tcaneb[row]) for col in range(1)] for row in range(len(tcaneb))]
            n = [" - ".join(head[i]) for i in range(len(head))]
            thor.removedTcaN = "By tca nebulosa criteria can be removed : " + " - ".join(n)
        pdf.cell(0, 5, thor.removedTcaN, 0, 1)
        filename = 'static/'+ datetime.now().strftime("%m_%d_%y_%H%M%S") + '.pdf'
        pdf.output( filename , 'F')

        return render_template('result.html', title="Result", thor=thor, tcan=tcan, file=filename)

    filename = 'static/'+ datetime.now().strftime("%m_%d_%y_%H%M%S") + '.pdf'
    pdf.output( filename , 'F')
    return render_template('result.html', title="Result", thor=thor, file=filename)