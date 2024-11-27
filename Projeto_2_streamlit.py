import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import statsmodels.api as sm
import statsmodels.formula.api as smf

#------------------------------# #-------------------------------# #-------------------------------#
sns.set(context='talk', style='ticks')

st.set_page_config(
    page_title="Previsão de renda",
    page_icon="favicon.ico",
    layout="wide"
)
#------------------------------# #-------------------------------# #-------------------------------#

# tabs configuration

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Inicio" 
                                        ,"Gráficos alteráveis" 
                                        ,"Univariadas" 
                                        ,"Bivariadas"
                                        ,"Dicionário"
                                        ,"Dataframe"]
                                        )

#------------------------------# #-------------------------------# #-------------------------------#
with tab1:
     # abrindo o arquivo
    uploaded_file = st.file_uploader("Escolha o arquivo CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

     # definindo uma mascara para outliers
    mascara = (df['renda'] <= 20000) & (df['renda'] >= 500)
    df = df[mascara]
     # rodando o modelo statsmodel
    reg = smf.ols('renda ~ sexo + posse_de_veiculo + posse_de_imovel + qtd_filhos + educacao + estado_civil + tipo_residencia + idade + tempo_emprego + qt_pessoas_residencia ''', data = df).fit()

    def predict_income(sexo, posse_de_veiculo, posse_de_imovel, qtd_filhos, educacao, estado_civil, tipo_residencia, idade, tempo_emprego, qt_pessoas_residencia):
        input_data = {
            'sexo': sexo,
            'posse_de_veiculo': posse_de_veiculo,
            'posse_de_imovel': posse_de_imovel,
            'qtd_filhos': qtd_filhos,
            'educacao': educacao,
            'estado_civil': estado_civil,
            'tipo_residencia': tipo_residencia,
            'idade': idade,
            'tempo_emprego': tempo_emprego,
            'qt_pessoas_residencia': qt_pessoas_residencia
        }

        # Crie um DataFrame a partir dos dados de entrada
        input_df = pd.DataFrame([input_data])

        # Faça a previsão usando seu modelo
        prediction = reg.predict(input_df)

        return prediction[0]

    # Interface do aplicativo usando Streamlit
        
    st.title('Análise exploratória da nossa :red[Previsão de Renda]')
    st.markdown('#### Nas abas seguintes, você encontrá informações, gráficos e métricas desenvolvidas.')
    st.markdown("##### O objetivo desta análise é prever a variação de renda dos clientes de uma instituição financeira. Os dados utilizados na análise e previsão foram coletados e distribuidos em variáveis com características diversas que auxiliam na previsão ou explicação da renda de um cliente.")
    st.markdown("---")

    st.markdown("\n" * 5)
    
#------------------------------# #-------------------------------# #-------------------------------#
with tab2:    

    #Gráfico customizável univariadas
    st.write('# Gráfico customizáveis para analise.')
    option1 = st.selectbox(
        'Selecione a variável',
        ('sexo', 'posse_de_veiculo', 'posse_de_imovel', 
         'qtd_filhos', 'tipo_renda', 'educacao', 
         'estado_civil', 'tipo_residencia', 'idade', 
         'tempo_emprego', 'qt_pessoas_residencia')
        )
    
    fig = plt.figure(figsize=(6, 4))
    sns.lineplot(x='data_ref',y='renda', hue= option1 ,data=df)
    plt.xlabel(option1)
    plt.xticks(rotation=45)
    st.pyplot(plt)

      
    st.divider()

    options = st.multiselect(
        'Selecione até duas variáveis',
        ['renda','sexo', 'posse_de_veiculo', 'posse_de_imovel', 
        'qtd_filhos', 'tipo_renda', 'educacao', 
        'estado_civil', 'tipo_residencia', 'idade', 
        'tempo_emprego', 'qt_pessoas_residencia'],
        ['educacao', 'renda']
        )
    
    st.write('# Gráfico customizado para variaveis bivariadas.')
    fig = plt.figure(figsize=(6, 4))
    sns.barplot(x=options[0],y=options[1],data=df)
    plt.xticks(rotation=30)
    st.pyplot(plt)

#------------------------------# #-------------------------------# #-------------------------------#
with tab3:

    #plots        
    st.write('# Gráficos ao decorrer do tempo')
    st.divider()

    # Criar subplots
    fig, ax = plt.subplots(6, 1, figsize=(10, 40), sharex=True, gridspec_kw={'hspace': 0.5})

    # Gráfico 1
    sns.lineplot(x='data_ref', y='renda', hue='posse_de_imovel', data=df, ax=ax[0])
    ax[0].tick_params(axis='x', rotation=45)
    ax[0].set_title('| Renda x Posse de Imóvel |')

    # Gráfico 2
    sns.lineplot(x='data_ref', y='renda', hue='posse_de_veiculo', data=df, ax=ax[1])
    ax[1].tick_params(axis='x', rotation=45)
    ax[1].set_title('| Renda x Posse de Veículo |')

    # Gráfico 3
    sns.lineplot(x='data_ref', y='renda', hue='qtd_filhos', data=df, ax=ax[2])
    ax[2].tick_params(axis='x', rotation=45)
    ax[2].set_title('| Renda x Quantidade de Filhos |')

    # Gráfico 4
    sns.lineplot(x='data_ref', y='renda', hue='tipo_renda', data=df, ax=ax[3])
    ax[3].tick_params(axis='x', rotation=45)
    ax[3].set_title('| Renda x Tipo de Renda |')

    # Gráfico 5
    sns.lineplot(x='data_ref', y='renda', hue='educacao', data=df, ax=ax[4])
    ax[4].tick_params(axis='x', rotation=45)
    ax[4].set_title('| Renda x Educação |')

    # Gráfico 6
    sns.lineplot(x='data_ref', y='renda', hue='estado_civil', data=df, ax=ax[5])
    ax[5].tick_params(axis='x', rotation=45)
    ax[5].set_title('| Renda x Estado Civil |')

    sns.despine()

    st.pyplot(fig)

 #------------------------------# #-------------------------------# #-------------------------------#    
with tab4: 

    st.write('# Gráficos das bivariadas.')
    st.divider()
    fig, ax = plt.subplots(7,1,figsize=(10,40))
    sns.barplot(x='posse_de_imovel',y='renda',data=df, ax=ax[0])
    sns.barplot(x='posse_de_veiculo',y='renda',data=df, ax=ax[1])
    sns.barplot(x='qtd_filhos',y='renda',data=df, ax=ax[2])
    sns.barplot(x='tipo_renda',y='renda',data=df, ax=ax[3])
    sns.barplot(x='educacao',y='renda',data=df, ax=ax[4])
    ax[4].tick_params(axis='x', rotation=20)
    sns.barplot(x='estado_civil',y='renda',data=df, ax=ax[5])
    sns.barplot(x='tipo_residencia',y='renda',data=df, ax=ax[6])
    ax[6].tick_params(axis='x', rotation=30)
    sns.despine()
    st.pyplot(plt)

#------------------------------# #-------------------------------# #-------------------------------#
with tab5:

    st.markdown("<h1 style='text-align: left; '>Dicionário dos dados</h1>", unsafe_allow_html=True)

    st.markdown(
        """
        | Variável                | Descrição                                           | Tipo         |
        | ----------------------- |:---------------------------------------------------:| ------------:|
        | data_ref                |  Data de referência da coleta do dado               | texto|
        | id_cliente              |  Número de identificação do cliente                 | inteiro|
        | sexo                    |  M = 'Masculino'; F = 'Feminino'                    | inteiro|
        | posse_de_veiculo        |  True = 'possui'; False = 'não possui'              | booleana|
        | posse_de_imovel         |  True = 'possui'; False = 'não possui'              | booleana|
        | qtd_filhos              |  Quantidade de filhos do cliente                    | inteiro|
        | tipo_renda              |  Tipo de renda (ex: assaliariado, autônomo etc)     | texto|
        | educacao                |  Nível educacional (ex: secundário, superior etc)   | texto|
        | estado_civil            |  Estado civil (ex: solteiro, casado etc)            | texto|
        | tipo_residencia         |  Tipo de residência (ex: casa/apartamento, com os pais etc)| texto|
        | idade                   |  Idade em anos                                      | inteiro|
        | tempo_emprego           |  Tempo de emprego em anos                           | float|
        | qt_pessoas_residencia   |  Quantidade de pessoas na residência                | float|
        | renda                   |  Valor da renda mensal                              | float|
    """
    )
    
    #------------------------------# #-------------------------------# #-------------------------------#
with tab6:   
    st.divider()
    st.markdown("<h1 style='text-align: center; '>Dataframe</h1>", unsafe_allow_html=True)
    st.write(df)
    #------------------------------# #-------------------------------# #-------------------------------#
