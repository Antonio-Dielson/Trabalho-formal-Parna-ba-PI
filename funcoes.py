from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def calcular_ql(e_ki, e_i, e_k, e):
    """Calcula o Quociente Locacional (QL)"""
    s_ki = e_ki / e_i
    s_i = e_k / e
    return s_ki / s_i

def calcular_ihh(e_ki, e_i, e_k, e):
    """Calcula o Índice de Herfindahl-Hirschman Modificado (IHH)"""
    s_ki = e_ki / e_k
    s_i = e_i / e
    return s_ki - s_i

def calcular_pr(e_ki, e_k):
    """Calcula a Participação Relativa (PR)"""
    return e_ki / e_k

def calcular_icn(ql, ihh, pr):
    """Calcula o Índice de Concentração Normalizado (ICN)"""
    # Padronizar os componentes
    scaler = StandardScaler()
    componentes = np.column_stack([ql, ihh, pr])
    componentes_padronizados = scaler.fit_transform(componentes)
    
    # PCA com 1 componente
    pca = PCA(n_components=1)
    pca.fit(componentes_padronizados)
    
    # Os pesos são as cargas do primeiro componente principal
    pesos = pca.components_[0]
    pesos_normalizados = pesos / pesos.sum()
    
    # Calcular ICN como média ponderada
    icn = (ql * pesos_normalizados[0] + 
           ihh * pesos_normalizados[1] + 
           pr * pesos_normalizados[2])
    
    return icn, pesos_normalizados

def carregar_dados(caminho_arquivo, nome_regiao):
    df = pd.read_csv(caminho_arquivo, encoding='latin1',sep=';')
    
    # Corrigir nomes dos setores
    novos_nomes = {
        '1 - Extrativa mineral': 'Extrativa_mineral',
        '2 - Indïŋ―stria de transformaïŋ―ïŋ―o': 'Industria_de_transformacao',
        '3 - Servicos industriais de utilidade pïŋ―blica': 'Servicos_industriais_de_utilidade_publica',
        '4 - Construïŋ―ïŋ―o Civil': 'Construcao_civil',
        '5 - Comïŋ―rcio': 'Comercio',
        '6 - Serviïŋ―os': 'Servicos',
        '7 - Administraïŋ―ïŋ―o Pïŋ―blica': 'Administracao_publica',
        '8 - Agropecuïŋ―ria, extraïŋ―ïŋ―o vegetal, caïŋ―a e pesca': 'Agropecuaria_extracao_vegetal_caca_e_pesca'
    }
    
    df['IBGE Setor'] = df['IBGE Setor'].replace(novos_nomes)
    df = df.set_index('IBGE Setor')
    df = df.drop(columns=['Total'])
    
    # Adicionar nome da região como coluna
    df['Regiao'] = nome_regiao
    
    return df