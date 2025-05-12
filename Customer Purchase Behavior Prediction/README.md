# Customer Purchase Prediction - Binary Classification

## 📌 Visão Geral

Este projeto apresenta um modelo de aprendizado de máquina criado no PyTorch para prever a probabilidade de um cliente fazer uma compra, com base em características demográficas e comportamentais. Ele aborda um problema de classificação binária usando um conjunto de dados disponível publicamente no Kaggle.

O modelo tem como objetivo ajudar as empresas a entender o comportamento do cliente e direcionar os possíveis compradores de forma mais eficaz.

## 📂 Dataset
- Fonte: [Kaggle - Predict Customer Purchase Behavior](https://www.kaggle.com/datasets/rabieelkharoua/predict-customer-purchase-behavior-dataset/data)
- Características:
  - Age;
  - Gender;
  - Annual Income;
  - Number of Purchases;
  - Product Category;
  - Time Spent on Website;
  - Loyalty Program;
  - Discounts Availed;
- Variável Alvo 🎯: Purchase Status

## 🛠️ Project Pipeline

O arquivo referente a etapa da EDA pode ser visto aqui: [eda.ipynb](https://github.com/AdEmanuel/PPGEEC2318/blob/main/Customer%20Purchase%20Behavior%20Prediction/eda.ipynb), ao passo que o arquivo 
contendo os passos descritos na etapa 2, 3 e 4 é o [purchase_classifier](https://github.com/AdEmanuel/PPGEEC2318/blob/main/Customer%20Purchase%20Behavior%20Prediction/purchase_classifier.ipynb).

### 1. Exploratory Data Analysis (EDA)
Durante a etapa de Análise Exploratória dos Dados, observou-se que a variável-alvo (_Purchase Status_) apresenta um desbalanceamento entre as classes, o que poderá influenciar negativamente o desempenho do modelo se não for tratado adequadamente. Além disso, verificou-se que as variáveis _Loyalty Program_ e _Discounts Availed_ demonstram maior correlação com a variável de saída, indicando um impacto maior que as outras no comportamento de compra dos clientes.

No que diz respeito às variáveis numéricas, foi identificado que estas operam em escalas bastante distintas — por exemplo, a variável _Annual Income_ varia aproximadamente entre 20.000 e 150.000, enquanto Age apresenta valores entre 18 e 70. Já entre as variáveis categóricas, a maioria apresenta duas categorias, com exceção da variável _Product Category_, que contém diversas classes distintas. Tais característica tornam necessário o emprego de técnicas de normalização dos dados e codificação categórica.

### 📚 2. Dados de avaliação

O conjunto de dados em estudo é dividido em Train e Test durante o estágio Segregate do pipeline de dados. 80% dos dados limpos são usados para treinar e os 20% restantes para testar. Além disso, 20% dos dados do Train são usados para fins de validação.

### 💪 3. Treinamento

*3.1 Preprocessing and Tensor Preparation*

Antes do treinamento, os dados foram processados utilizando pipelines do Scikit-learn, com codificação one-hot aplicada às variáveis categóricas e normalização padronizada para as variáveis numéricas. Em seguida, a técnica SMOTE foi utilizada para balancear a variável alvo na base de treino. Os dados resultantes foram convertidos em tensores utilizando o PyTorch, sendo organizados em TensorDatasets e carregados em mini-lotes com o auxílio da classe DataLoader.

*3.2 Model*

O modelo treinado é uma rede neural simples configurada para tarefa de classificação binária. Sua estrutura está encapsulada na classe LogisticRegressionModel. A função de perda utilizada foi BCEWithLogitsLoss e o otimizador escolhido foi o Adam, com taxa de aprendizado de 0.01.

*3.3 Training Framework*

A classe _Architecture_ foi responsável por gerenciar todo o ciclo de treinamento, validação e checkpointing. Ela centraliza as operações de propagação direta, retropropagação, atualização dos pesos e cálculo das métricas, operando diretamente sobre tensores e garantindo compatibilidade com GPU. A função train() foi executada por 100 épocas.

### 📊 4. Função de Perda (Loss Function) e Métricas de Desempenho

A curva de perda evidencia uma boa convergência ao longo das 100 épocas e sem ocorrência de overfitting relevante, dado que as curvas de treino e validação permanecem próximas.

Quanto às métricas de desempenho, o modelo obteve Accuracy de 85,14%, Precision de 85,86%, Recall de 81,73% e F1-Score de 83,74%, indicando um bom equilíbrio entre _precision_ e _sensitivity_. A área sob a curva ROC (AUC = 0.8970) e o formato da curva Precision-Recall também demonstram boa capacidade discriminativa do classificador.

## Referências

- [Kaggle - Predict Customer Purchase Behavior](https://www.kaggle.com/datasets/rabieelkharoua/predict-customer-purchase-behavior-dataset/data)
- Repositório do Prof Dr. Ivanovitch [Link](https://github.com/ivanovitchm/PPGEEC2318)

## Colaboradores
- Adson Emanuel
- Klyfton Stanley

