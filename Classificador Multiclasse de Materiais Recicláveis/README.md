# Classificador de Materiais Recicláveis

## 📌 Visão Geral

Este projeto apresenta uma solução de aprendizado de máquina, implementada em PyTorch, para realizar a classificação automática de imagens de resíduos recicláveis em quatro classes distintas: glass, metal, paper e plastic. Trata-se de um problema de classificação multiclasse, com foco em redes neurais convolucionais (CNNs) aplicadas à Visão Computacional.



A arquitetura do modelo foi encapsulada na classe Architecture, responsável por organizar todo o pipeline de pré-processamento, treinamento, validação e inferência, com suporte a execução em GPU.



O projeto foi desenvolvido como parte da avaliação final da disciplina PPGEEC2318 - Machine Learning ministrada pelo Prof. Dr. Ivanovitch Medeiros, no Programa de Pós-Graduação em Engenharia Elétrica e de Computação da UFRN



## 📂 Dataset



O conjunto de dados utilizado é uma adaptação do TrashNet: A set of annotated images of trash that can be used for object detection Dataset, desenvolvido pelo Polygence Project e disponibilizado na plataforma Roboflow.



Embora o dataset original contenha seis classes (cardboard, glass, metal, paper, plastic e trash), este projeto considera apenas as quatro categorias relacionadas à coleta seletiva: glass, metal, paper e plastic.



Foram utilizadas 400 imagens de treinamento e 100 imagens de validação para cada classe, em que cada uma delas tem 512x384 pixels. 



## Descrição do Modelo

## Resultados

## Conclusão

## 🔗 Referências

* [Roboflow - trashnet Computer Vision Project](https://universe.roboflow.com/myspace-uc4uq/trashnet-sn7pu)
* [Repositório do Prof Dr. Ivanovitch](https://github.com/ivanovitchm/PPGEEC2318)

## 👥 Colaboradores

* Adson Emanuel
* Klyfton Stanley
