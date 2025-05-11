# Customer Purchase Behavior Prediction - Binary Classification

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

### Exploratory Data Analysis (EDA)
Nesta etapa foi possível observar que os dados referentes a variável alvo estão desbalanceados. Além disso, notou-se que as _features_ _Loyalt Program_ e _Discounts Availed_ são as que mais se correlacionam com _Purcahse Status_
