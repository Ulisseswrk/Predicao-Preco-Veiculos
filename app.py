import pickle
import pandas as pd
import streamlit as st

# Configuração da página
st.set_page_config(
    page_title="Predição de Preço de Veículos",
    layout="wide",
)

# Título principal
st.title("Simulador Preditivo de Preço de Veículos")
st.markdown("Insira as características do veículo para obter a estimativa de preço gerada pelo modelo de regressão múltipla.")

# Range de ano de fabricação presente na base usada para treinar o modelo
ANO_MIN_TREINO = 2000
ANO_MAX_TREINO = 2023

# Tamanhos de motor presentes na base usada para treinar o modelo (1.0 a 5.0, passo 0.1)
TAMANHOS_MOTOR = [round(1.0 + 0.1 * i, 1) for i in range(41)]

# Entradas do usuário para o modelo
col1, col2 = st.columns(2)

with col2:
    tipo_combustivel = st.selectbox("Tipo de Combustível", ["Gasolina", "Diesel", "Hibrido", "Eletrico"])
    cambio = st.selectbox("Tipo de Câmbio", ["Manual", "Automático", "Semiautomático"])

with col1:
    ano = st.number_input("Ano de Fabricação", min_value=1990, max_value=2026, value=2020)
    quilometragem = st.number_input("Quilometragem (km)", min_value=0, max_value=500000, value=30000, step=1000)
    if tipo_combustivel == "Eletrico":
        motor = 2.0
    else:
        motor = st.selectbox("Tamanho do Motor (L)", TAMANHOS_MOTOR, index=TAMANHOS_MOTOR.index(2.0))

if ano < ANO_MIN_TREINO or ano > ANO_MAX_TREINO:
    st.warning(f"Atenção: o ano {ano} está fora do range de {ANO_MIN_TREINO} a {ANO_MAX_TREINO} usado para treinar o modelo. A predição pode ser pouco confiável.")

# Trava: veículos elétricos só existem com câmbio automático
combinacao_invalida = tipo_combustivel == "Eletrico" and cambio != "Automático"
if combinacao_invalida:
    st.warning("Veículos elétricos só existem com câmbio automático. Selecione 'Automático'.")

# Botão para realizar a predição
if st.button("Calcular Preço Estimado", disabled=combinacao_invalida):
    try:
        with open('model/modelo_regressao_multipla.pkl', 'rb') as f:
            modelo = pickle.load(f)
        
        # Criar dataframe com os inputs do usuário na mesma estrutura do X_train
        input_usuario = pd.DataFrame([[ano, motor, tipo_combustivel, cambio, quilometragem]],
                                     columns=['Ano', 'Motor', 'Tipo_Combustivel', 'Cambio', 'Quilometragem'])
        
        predicao = modelo.predict(input_usuario)
        
        # A função .item() força a extração do valor numérico puro de dentro do array do NumPy
        valor_estimado = float(predicao.item())
        
        st.success(f"O preço estimado para o veículo é de: R$ {valor_estimado:,.2f}")
        
    except FileNotFoundError:
        st.error("Erro: O arquivo `modelo_regressao_multipla.pkl` não foi encontrado na pasta `model/`. Verifique o caminho.")
    except Exception as e:
        st.error(f"Ocorreu um erro durante a execução: {e}")