from sklearn.tree import DecisionTreeClassifier
import numpy as np
import streamlit as st

# Tempo de uso do produto x número de reclamações
x = np.array([
    [1, 5],
    [2, 4],
    [3, 3],
    [4, 1],
    [4, 1],
    [5, 0]
])

# 0 = fica | 1 = cancela
y = np.array([0, 1, 1, 0, 1, 1])

# Criando e treinando o modelo
modelo = DecisionTreeClassifier()
modelo.fit(x, y)

# Entrada de dados
st.title("Previsão de cancelamento")

uso = st.number_input(
    "Quantidade de vezes que o produto foi utilizado:",
    min_value=0,
    value=0
)

reclamacoes = st.number_input(
    "Quantidade de reclamações:",
    min_value=0,
    value=0
)

# Fazer a previsão
if st.button("Analisar1"):
    resultado = modelo.predict([[uso, reclamacoes]])

    if resultado[0] == 1:
        st.write("O cliente provavelmente vai cancelar.")
    else:
        st.write("O cliente provavelmente vai continuar.")


# analise 

df['previsao_modelo'] = modelo.predict(X)

st.subheader('analise completa')


df['status'] = ['CLIENTE CONSOLIDADO' if p == 0 else 'POSSÍVEL CANCELAMENTO' for p in modelo.predict(X)]
st.dataframe(df)

df.to_html('dados.html')

# print(modelo.predict([[5,2]]))
# print(modelo.predict([[10,2]]))
# print(modelo.predict([[3,0]]))
# print(modelo.predict([[7,2]]))
# print(modelo.predict([[1,0]]))