import pickle
import pandas as pd
import streamlit as st

# Configuração da página
st.set_page_config(
    page_title="Predição de Preço de Veículos",
    page_icon="🚗",
    layout="wide",
)

# Título principal
st.title("Simulador Preditivo de Preço de Veículos")
st.markdown("Insira as características do veículo para obter a estimativa de preço gerada pelo modelo de regressão múltipla.")

# Entradas do usuário para o modelo
col1, col2 = st.columns(2)

with col1:
    ano = st.number_input("Ano de Fabricação", min_value=1990, max_value=2026, value=2020)
    quilometragem = st.number_input("Quilometragem (km)", min_value=0, max_value=500000, value=30000)
    motor = st.number_input("Tamanho do Motor (L)", min_value=1.0, max_value=6.0, value=2.0, step=0.1)

with col2:
    tipo_combustivel = st.selectbox("Tipo de Combustível", ["Gasolina", "Diesel", "Hibrido", "Eletrico"])
    cambio = st.selectbox("Tipo de Câmbio", ["Manual", "Automático", "Semiautomático"])

# Botão para realizar a predição
if st.button("Calcular Preço Estimado"):
    try:
        with open(r'model\modelo_regressao_multipla.pkl', 'rb') as f:
            modelo = pickle.load(f)
        
        # Criar dataframe com os inputs do usuário na mesma estrutura do X_train
        input_usuario = pd.DataFrame([[ano, motor, tipo_combustivel, cambio, quilometragem]],
                                     columns=['Ano', 'Motor', 'Tipo_Combustivel', 'Cambio', 'Quilometragem'])
        
        predicao = modelo.predict(input_usuario)
        
        # A função .item() força a extração do valor numérico puro de dentro do array do NumPy
        valor_estimado = float(predicao.item())
        
        st.success(f"O preço estimado para o veículo é de: R$ {valor_estimado:,.2f}")
        
    except FileNotFoundError:
        st.error(r"Erro: O arquivo `modelo_regressao_multipla.pkl` não foi encontrado na pasta `model\`. Verifique o caminho.")
    except Exception as e:
        st.error(f"Ocorreu um erro durante a execução: {e}")