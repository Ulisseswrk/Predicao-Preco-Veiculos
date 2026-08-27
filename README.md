# Predição de Preço de Veículos

Projeto de Data Science desenvolvido como Checkpoint 4 (Regressão Linear) da disciplina **Data Science & Statistical Computing** — FIAP, 2026.

O projeto compara diferentes modelos de regressão (referência, simples, múltipla e polinomial) para estimar o **preço de venda de um veículo** a partir de suas características (ano, motor, quilometragem, combustível e câmbio), e expõe o modelo vencedor em uma **interface web interativa feita em Streamlit**.

---

## Sumário

- [Sobre o problema](#sobre-o-problema)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como funciona a solução](#como-funciona-a-solução)
  - [1. Base de dados](#1-base-de-dados)
  - [2. Limpeza e tratamento](#2-limpeza-e-tratamento)
  - [3. Análise exploratória](#3-análise-exploratória)
  - [4. Modelagem](#4-modelagem)
  - [5. Resultados comparativos](#5-resultados-comparativos)
  - [6. Modelo final e diagnóstico](#6-modelo-final-e-diagnóstico)
  - [7. App Streamlit](#7-app-streamlit)
- [Pré-requisitos](#pré-requisitos)
- [Passo a passo para rodar o projeto](#passo-a-passo-para-rodar-o-projeto)
  - [1. Clonar o repositório](#1-clonar-o-repositório)
  - [2. Criar e ativar um ambiente virtual](#2-criar-e-ativar-um-ambiente-virtual)
  - [3. Instalar as dependências](#3-instalar-as-dependências)
  - [4. Rodar o notebook (opcional — retreinar o modelo)](#4-rodar-o-notebook-opcional--retreinar-o-modelo)
  - [5. Rodar a aplicação Streamlit](#5-rodar-a-aplicação-streamlit)
- [Usando o simulador](#usando-o-simulador)
- [Solução de problemas (Troubleshooting)](#solução-de-problemas-troubleshooting)
- [Limitações conhecidas](#limitações-conhecidas)
- [Licença](#licença)

---

## Sobre o problema

**Pergunta de pesquisa:** em que medida o ano de fabricação, a quilometragem rodada e a litragem do motor ajudam a prever o preço de venda de um veículo no mercado?

O mercado de veículos seminovos/usados sofre com assimetria de informação entre compradores e vendedores. Um modelo preditivo que estima o valor justo de um carro com base no seu desgaste (quilometragem/ano) e especificações técnicas protege consumidores de preços abusivos e ajuda lojistas a precificar seus estoques de forma orientada a dados.

- **Variável resposta (y):** `Preco` (numérica contínua, em R$)
- **Variáveis explicativas (X):** `Ano`, `Quilometragem`, `Motor`, `Cambio`, `Tipo_Combustivel`

---

## Estrutura do repositório

```
predicao-preco-veiculos/
├── app.py                              # Aplicação web (Streamlit) que consome o modelo treinado
├── notebook.ipynb                      # Notebook com todo o pipeline de Data Science (EDA + modelagem)
├── requirements.txt                    # Dependências Python do projeto
├── data/
│   └── base.csv                        # Base de dados bruta (10.000 registros, em inglês/USD)
├── model/
│   └── modelo_regressao_multipla.pkl   # Modelo (Pipeline sklearn) treinado e serializado
├── LICENSE                             # Licença Apache 2.0
└── README.md
```

---

## Como funciona a solução

Todo o raciocínio de ciência de dados está documentado célula a célula em [`notebook.ipynb`](notebook.ipynb). Resumo do pipeline:

### 1. Base de dados

- Fonte: [Kaggle — Car Price Dataset](https://www.kaggle.com/datasets/mos3santos/conjunto-de-dados-de-preos-de-carros), carregada de [`data/base.csv`](data/base.csv).
- 10.000 registros, colunas originais em inglês: `Brand`, `Model`, `Year`, `Engine_Size`, `Fuel_Type`, `Transmission`, `Mileage`, `Doors`, `Owner_Count`, `Price` (em USD).
- O notebook consome a API pública [open.er-api.com](https://open.er-api.com) para pegar a **cotação atual do dólar** e converter a coluna `Price` para reais (BRL) em tempo de execução — ou seja, **os valores de preço mudam a cada vez que o notebook é reexecutado**, de acordo com a cotação do dia.

### 2. Limpeza e tratamento

- Colunas renomeadas para português (`Ano`, `Modelo`, `Motor`, `Tipo_Combustivel`, `Cambio`, `Quilometragem`, `Quantidade_Portas`, `Quantidade_Donos`, `Preco`).
- Descartadas: `Marca` e `Modelo` (alta cardinalidade, risco de overfitting) e `Quantidade_Portas`/`Quantidade_Donos` (baixa variabilidade/relevância).
- Categorias traduzidas: `Fuel_Type` → `Diesel/Hibrido/Eletrico/Gasolina`; `Transmission` → `Manual/Automático/Semiautomático`.
- Verificação de nulos e duplicados: **nenhum encontrado**.

Dicionário de dados final (6 colunas ativas):

| Variável | Tipo | Categorias/Unidade | Papel |
| :--- | :--- | :--- | :--- |
| Ano | Numérica discreta | Anos (ex: 2020) | Explicativa |
| Motor | Numérica contínua | Litros (ex: 1.0–6.0) | Explicativa |
| Tipo_Combustivel | Categórica nominal | Gasolina, Diesel, Hibrido, Eletrico | Explicativa |
| Cambio | Categórica nominal | Manual, Automático, Semiautomático | Explicativa |
| Quilometragem | Numérica contínua | Km | Explicativa |
| Preco | Numérica contínua | R$ | Resposta (alvo) |

### 3. Análise exploratória

- Distribuição do preço: simétrica, sem assimetria extrema.
- Boxplot: ausência de outliers — mantidas as 10.000 observações.
- Dispersão Quilometragem × Preço: relação inversamente proporcional (depreciação por uso).
- Matriz de correlação: orienta a escolha do preditor único (`Ano`) usado na regressão simples.

### 4. Modelagem

Divisão treino/teste: 80/20 (`random_state=42`). Pré-processamento via `ColumnTransformer` (`OneHotEncoder` para categóricas + `StandardScaler` para numéricas), tudo dentro de um `Pipeline` do scikit-learn. Quatro abordagens comparadas:

1. **Modelo de referência** — sempre prevê a média do preço de treino.
2. **Regressão Linear Simples** — usa apenas `Ano` como preditor.
3. **Regressão Linear Múltipla** — usa todas as variáveis explicativas (`Ano`, `Motor`, `Tipo_Combustivel`, `Cambio`, `Quilometragem`) via pipeline com encoding/scaling. **Este é o modelo salvo em produção.**
4. **Regressão Polinomial (grau 2 em Quilometragem)** — testa se uma relação não linear melhora o ajuste.

### 5. Resultados comparativos

| Modelo | MAE | RMSE | R² |
| :--- | ---: | ---: | ---: |
| Referência (média) | 12.715,49 | 15.623,83 | -0,0016 |
| Regressão Simples | 9.527,53 | 11.704,90 | 0,4378 |
| **Regressão Múltipla** | **99,84** | **332,52** | **0,9995** |
| Regressão Polinomial | 102,46 | 332,60 | 0,9995 |

A **Regressão Linear Múltipla** foi escolhida como modelo final: usando todas as variáveis, o modelo praticamente reconstrói a fórmula sintética usada para gerar o dataset, e a complexidade extra da versão polinomial não trouxe ganho que a justifique.

> A base de dados aparenta ser **sintética** — por isso o erro da regressão múltipla é tão baixo (R² ≈ 0,9995). Em dados reais de mercado, espera-se um erro consideravelmente maior, já que fatores como estado de conservação, histórico de acidentes e revisões não estão presentes na base.

### 6. Modelo final e diagnóstico

Sobre a Regressão Múltipla, o notebook realiza diagnóstico de resíduos (real × previsto, resíduos × previstos, distribuição dos resíduos, QQ-plot) e uma matriz de correlação entre as variáveis numéricas para confirmar ausência de multicolinearidade relevante. Ao final, o pipeline treinado (pré-processamento + regressão) é serializado com `pickle` em [`model/modelo_regressao_multipla.pkl`](model/modelo_regressao_multipla.pkl).

**Atenção:** o modelo foi treinado com carros com ano **entre 2000 e 2024**. Extrapolar para anos muito fora dessa faixa pode gerar previsões irreais (negativas ou absurdamente altas).

### 7. App Streamlit

O arquivo [`app.py`](app.py) carrega o `.pkl` e oferece um formulário simples (ano, quilometragem, motor, combustível, câmbio) que monta um DataFrame no mesmo formato usado no treino e chama `modelo.predict(...)` para exibir o preço estimado.

---

## Pré-requisitos

- **Python 3.10+** instalado (o projeto foi testado com Python 3.11).
- **Git** (para clonar o repositório).
- Conexão com a internet **apenas se for reexecutar o notebook do zero** (ele consulta uma API de cotação de câmbio). Para apenas rodar o app Streamlit com o modelo já treinado, internet não é necessária.

Verifique sua versão do Python:

**PowerShell (Windows):**
```powershell
python --version
```

**Bash (Linux/macOS):**
```bash
python3 --version
```

---

## Passo a passo para rodar o projeto

### 1. Clonar o repositório

```powershell
git clone https://github.com/<seu-usuario>/predicao-preco-veiculos.git
cd predicao-preco-veiculos
```

Se você já tem a pasta localmente, apenas abra um terminal dentro dela.

### 2. Criar e ativar um ambiente virtual

É fortemente recomendado usar um ambiente virtual para não misturar as dependências deste projeto com outras instalações Python.

**PowerShell (Windows):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> Se o PowerShell bloquear a execução do script de ativação com um erro de política de execução, rode uma vez (na mesma sessão do terminal):
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> ```

**Prompt de Comando (cmd.exe):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Bash (Linux/macOS):**
```bash
python3 -m venv venv
source venv/bin/activate
```

Depois de ativado, o prompt do terminal deve exibir `(venv)` no início da linha.

### 3. Instalar as dependências

Com o ambiente virtual ativo:

```powershell
pip install -r requirements.txt
```

Isso instala: `requests`, `scipy`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` e `streamlit`.

Para rodar o notebook também é necessário o Jupyter (não listado no `requirements.txt`, pois não é obrigatório para o app):

```powershell
pip install notebook jupyterlab ipykernel
```

### 4. Rodar o notebook (opcional — retreinar o modelo)

O repositório já vem com um modelo treinado em `model/modelo_regressao_multipla.pkl`, então **este passo é opcional** — só é necessário se você quiser reexecutar a análise, gerar os gráficos ou retreinar o modelo (por exemplo, com uma cotação de dólar mais atual).

```powershell
jupyter notebook notebook.ipynb
```
ou, se preferir o Jupyter Lab:
```powershell
jupyter lab
```

Isso abrirá o notebook no navegador. Execute as células **em ordem, de cima para baixo** (`Kernel > Restart & Run All` é o caminho mais seguro). A célula de leitura de câmbio depende de internet; se a API estiver indisponível, ajuste manualmente a variável `cotacao` para continuar a execução.

Ao final da execução, a última célula regrava o arquivo `model/modelo_regressao_multipla.pkl` com o novo modelo treinado.

Alternativa via VS Code: abra `notebook.ipynb` na extensão Jupyter do VS Code e use "Run All".

### 5. Rodar a aplicação Streamlit

Com o ambiente virtual ativo e as dependências instaladas, a partir da **raiz do projeto** (onde está o `app.py`):

```powershell
streamlit run app.py
```

O terminal exibirá algo como:
```
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

O navegador padrão deve abrir automaticamente. Se não abrir, acesse manualmente `http://localhost:8501`.

Para encerrar o servidor, volte ao terminal e pressione `Ctrl + C`.

---

## Usando o simulador

Na interface web, preencha os campos:

- **Ano de Fabricação** (1990–2026, recomendado manter entre 2000–2024, faixa em que o modelo foi treinado)
- **Quilometragem (km)**
- **Tamanho do Motor (L)**
- **Tipo de Combustível** (Gasolina, Diesel, Híbrido, Elétrico)
- **Tipo de Câmbio** (Manual, Automático, Semiautomático)

Clique em **"Calcular Preço Estimado"** para ver o valor previsto pelo modelo de Regressão Múltipla.

---

## Solução de problemas (Troubleshooting)

**`streamlit: command not found` / "streamlit não é reconhecido"**
O ambiente virtual não está ativado ou as dependências não foram instaladas. Repita os passos 2 e 3.

**Erro `FileNotFoundError` sobre `modelo_regressao_multipla.pkl`**
Rode o `streamlit run app.py` a partir da **raiz do repositório** (o app usa o caminho relativo `model\modelo_regressao_multipla.pkl`). Se o arquivo `.pkl` não existir, gere-o executando o notebook completo (passo 4).

**Erro ao ativar o venv no PowerShell (`... não pode ser carregado porque a execução de scripts foi desabilitada`)**
Rode `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` no mesmo terminal antes de ativar o venv novamente.

**Erro de conexão ao rodar a célula da cotação do dólar no notebook**
A API pública `open.er-api.com` pode estar temporariamente fora do ar ou sem acesso à internet. Substitua manualmente o valor de `cotacao` na célula (ex: `cotacao = 5.15`) e continue a execução das células seguintes.

**Versão incompatível de alguma biblioteca**
Se o `pip install -r requirements.txt` falhar por conflito de versões, tente atualizar o `pip` antes: `python -m pip install --upgrade pip`.

---

## Limitações conhecidas

- A base de dados aparenta ser **sintética**, o que explica o R² extremamente alto (≈0,9995) da regressão múltipla — não é indicativo direto de desempenho em dados reais de mercado.
- O modelo não deve ser usado para carros de luxo extremos, colecionáveis (raros/antigos) ou veículos fortemente customizados, pois fogem à regra geral de depreciação capturada pelo modelo.
- A base não contempla fatores relevantes do mundo real, como estado de conservação da lataria, histórico de sinistros/acidentes e revisões em concessionária.
- Extrapolar previsões para anos fora do intervalo de treino (2000–2024) pode gerar resultados irreais.

---

## Licença

Este projeto está licenciado sob os termos da [Apache License 2.0](LICENSE).
