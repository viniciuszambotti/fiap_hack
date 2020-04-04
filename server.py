from flask import Flask
from flask import request

import pandas as pd
import numpy as np
import pickle

from notebooks.classes.ClusterModel import ClusterModel

import json
app = Flask(__name__)


@app.route('/')
def hello_world():
    user = request.args.get('user') 

    modelo_clientes = pickle.load(open('models/kmeans_clientes_FINAL.pkl', 'rb'))
    modelo_fixa = pickle.load(open('models/recommendation_fixa_FINAL.pkl', 'rb'))
    modelo_fundos = pickle.load(open('models/recommendation_fundos_FINAL.pkl', 'rb'))
    clientes = pd.read_csv('data/processados/clientes_final.csv')
    fixa = pd.read_csv('data/processados/produto_fixa_final.csv')
    fundos = pd.read_csv('data/processados/produto_fundos_final.csv')
    full_fundos = pd.read_csv('data/processados/produtos_fundos.csv')
    full_fixa = pd.read_csv('data/processados/produtos_fixa.csv')
    jsn = json.loads(user)
    data = pd.DataFrame([jsn], columns=jsn.keys())
    cluster = clientes[clientes['Id'] == data['Id'].values[0]]['cluster']
    recomendacao_fixa = modelo_fixa.query(f'customerId == {cluster.values[0]}').iloc[0]['produtoId']
    recomendacao_fundos = modelo_fundos.query(f'customerId == {cluster.values[0]}').iloc[0]['produtoId']

    full_fundos = full_fundos[['ProdutoId', 'NomeInvestimento__c', 'FiltroValorMinimo__c']]
    full_fixa = full_fixa[['ProdutoId', 'NomeInvestimento__c', 'NomeProduto__c']]

    produtos_fixa = fixa.query(f'cluster == {recomendacao_fixa}').iloc[range(0,2)]
    produtos_fundos = fundos.query(f'cluster == {recomendacao_fundos}').iloc[range(0,2)]

    fundo =json.dumps({"fundo":produtos_fundos.merge(full_fundos, left_on='ProdutoId', right_on='ProdutoId', how='left').to_dict('records')})
    fixa =json.dumps({"fixa":produtos_fixa.merge(full_fixa, left_on='ProdutoId', right_on='ProdutoId', how='left').to_dict('records')})
    
    st = f'"data":[{fixa},{fundo}]'

    result = '{' + st + '}'
    return result

