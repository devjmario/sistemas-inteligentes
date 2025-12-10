Detecção de Fraudes em Transações Financeiras

Projeto para treinar um modelo de Machine Learning usando a base Credit Card Fraud Detection e simular transações para validação do modelo.

Como executar
1. Extraia os dados

Extraia o arquivo:

creditcard.rar → creditcard.csv

2. Treine o modelo
python treinar.py


Isso gera o arquivo modelo_fraude.pkl.

3. Rode o simulador
python simulador.py


Permite:

Inserir transações manualmente

Gerar transações legítimas ou fraudulentas

Classificação automática usando o modelo treinado

Arquivos

treinar.py → treina e salva o modelo

simulador.py → simula transações e aplica o modelo

creditcard.csv → base de dados (após extração)
