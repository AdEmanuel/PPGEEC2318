# Classificador de Materiais Recicláveis

## 📌 Visão Geral

Este projeto apresenta uma solução de aprendizado de máquina, implementada em PyTorch, para realizar a classificação automática de imagens de resíduos recicláveis em quatro classes distintas: glass, metal, paper e plastic. Trata-se de um problema de classificação multiclasse, com foco em redes neurais convolucionais (CNNs) aplicadas à Visão Computacional.

A arquitetura do modelo foi encapsulada na classe Architecture, responsável por organizar todo o pipeline de pré-processamento, treinamento, validação e inferência, com suporte a execução em GPU.

O projeto foi desenvolvido como parte da avaliação final da disciplina PPGEEC2318 - Machine Learning ministrada pelo Prof. Dr. Ivanovitch Medeiros, do Programa de Pós-Graduação em Engenharia Elétrica e de Computação da UFRN.

## 📂 Dataset

O conjunto de dados utilizado é uma adaptação do TrashNet: A set of annotated images of trash that can be used for object detection Dataset, desenvolvido pelo Polygence Project e disponibilizado na plataforma Roboflow.

Embora o dataset original contenha seis classes (cardboard, glass, metal, paper, plastic e trash), este projeto considera apenas as quatro categorias relacionadas à coleta seletiva: glass, metal, paper e plastic.

Foram utilizadas 400 imagens de treinamento e 100 imagens de validação para cada classe, em que cada uma delas tem 512x384 pixels. 

## Arquitetura e Desenvolvimento dos Modelos

A metodologia utilizada consistiu na implementação de dois modelos CNN (modelo base e modelo pessoal), buscando observar como alterações na arquitetura e no learning rate afetam no desempenho da classificação.

### Modelo Base

Este modelo, implementado no arquivo ClassifierModelBase.ipynb, utiliza uma arquitetura baseada no material de aula disponibilizado pelo professor. No pré-processamento, as imagens passaram apenas pelas transformações essenciais de redimensionamento (para o tamanho esperado pela rede) e conversão para o formato de tensor PyTorch.

A arquitetura da rede consiste em uma CNN sequencial com a seguinte estrutura:

- Bloco Convolucional 1: Uma camada Conv2d com 16 filtros, seguida por uma função de ativação ReLU e uma camada de MaxPool2d.
- Bloco Convolucional 2: Uma camada Conv2d com 32 filtros, também seguida por ReLU e MaxPool2d.
- Classificador: Duas camadas lineares (Linear) para realizar a classificação final nas quatro categorias.

Para o treinamento, utilizou-se a função de perda Cross-Entropy Loss (nn.CrossEntropyLoss) e o otimizador Adam com uma taxa de aprendizado (learning rate) de 3e-4, ao longo de 10 épocas.

-> 🔍 Visualizações: Filtros e Hooks

Para entender o comportamento interno da rede, foram utilizados filtros e hooks. Os filtros da primeira camada convolucional (conv1), foram visualizados para inspecionar os tipos de características que o modelo aprendia a detectar nos estágios iniciais (ex: bordas, texturas e padrões simples). Ao passo que os filtros da segunda camada (conv2) aprendem a combinar essas características simples para identificar padrões mais complexos e abstratos, como texturas específicas de cada material ou formas mais definidas.

[IMAGEM DOS FILTROS DA CAMADA CONV1 DO MODELO BASE]
Figura: Visualização dos 5 filtros da primeira camada convolucional do Modelo Base.

[IMAGEM DOS FILTROS DA CAMADA CONV1 DO MODELO BASE]
Figura: Visualização dos 5 filtros da segunda camada convolucional do Modelo Base.

Já os hooks foram utilizados para visualizar a transformação das imagens ao longo das camadas convolucionais, permitindo compreender o que cada camada aprende e como os filtros atuam sobre os dados. A seguir, são apresentados os mapas de características (feature maps) extraídos das camadas do featurizer (conv1, conv2) e do classifier (fc1, fc2).

### Modelo Pessoal

Partindo da análise do modelo anterior, foi desenvolvido o ClassifierPersonalModel.ipynb. Este modelo representa aplica alterações na preparação dos dados e na arquitetura da rede com o objetivo de construir uma rede mais robusta, capaz de aprender características mais detalhadas das imagens.

As principais modificações introduzidas neste modelo foram:

- Aumento da Resolução da Imagem: O tamanho das imagens de entrada foi alterado de 28x28 para 128x128 pixels. Essa mudança é fundamental, pois imagens com maior resolução contêm mais detalhes visuais. Para um problema de classificação de materiais, onde texturas sutis e padrões finos são importantes para a diferenciação (como o brilho do vidro ou a rugosidade do papel), fornecer mais pixels à rede permite que as camadas convolucionais extraiam características mais ricas e discriminativas, potencializando a precisão do modelo.

- Preparação de Dados com Data Augmentation: Para aumentar a robustez e a capacidade de generalização do modelo, foram aplicadas transformações aleatórias nas imagens de treinamento, como RandomHorizontalFlip (espelhamento horizontal), RandomRotation (rotações) e ColorJitter (alterações de brilho, contraste e saturação).

- Aumento da Complexidade da Arquitetura: O número de filtros nas camadas convolucionais foi expandido progressivamente para permitir que a rede aprendesse padrões mais complexos a partir dos dados de maior resolução. A arquitetura conta com 3 camadas convolucionais, sendo a primeira com n_feature filtros, a segunda com n_feature * 2 filtros e a terceira com n_feature * 4 filtros. Essa configuração amplia significativamente a capacidade da rede de extrair representações hierárquicas dos dados de entrada.

- Adição de Camadas de Regularização e Estabilização: Para gerenciar a maior complexidade da rede e mitigar o risco de overfitting, foram adicionadas camadas de BatchNorm2d após cada convolução para estabilizar o treinamento, e camadas de Dropout nas etapas finais do classificador.

## Resultados

## Conclusão

## 🔗 Referências

* [Roboflow - trashnet Computer Vision Project](https://universe.roboflow.com/myspace-uc4uq/trashnet-sn7pu)
* [Repositório do Prof Dr. Ivanovitch](https://github.com/ivanovitchm/PPGEEC2318)

## 👥 Colaboradores

* Adson Emanuel
* Klyfton Stanley
