# Projeto: Previsão de Risco de Inadimplência

Este projeto de Data Science constrói um modelo de *credit scoring* para prever a probabilidade de um cliente se tornar inadimplente, utilizando dados históricos para otimizar a concessão de crédito e minimizar perdas financeiras.

* **Fonte de Dados:** [Give Me Some Credit :: 2011 Competition Data](https://www.kaggle.com/c/GiveMeSomeCredit) do Kaggle.

---

## Metodologia

O projeto foi estruturado seguindo as principais etapas de um ciclo de vida de Machine Learning:

### 1. Carga e Armazenamento dos Dados
O processo inicia com a execução de um script Python (`01_carga_sql.py`) que realiza a carga inicial dos dados. Este script lê o arquivo `cs-training.csv`, realiza uma limpeza básica e armazena os dados de forma estruturada em um banco de dados **SQLite** (`analise_de_credito.db`). Essa abordagem garante um acesso mais eficiente e organizado aos dados para a fase de modelagem.

### 2. Análise Exploratória de Dados (EDA)
Com os dados no banco, a análise exploratória no Jupyter Notebook revelou insights cruciais:
* **Desbalanceamento de Classes:** A base de dados é fortemente desbalanceada, com apenas 7% dos clientes classificados como inadimplentes.
* **Principais Fatores de Risco:** Histórico de atrasos em pagamentos e a idade do cliente mostraram ser os indicadores mais correlacionados com a inadimplência.
* **Multicolinearidade:** Foi identificada uma alta correlação entre as variáveis de atraso (`atraso_30_59_dias`, `atraso_60_89_dias`, etc.).

### 3. Engenharia e Pré-processamento de Features
Para otimizar o desempenho do modelo, foram aplicadas as seguintes técnicas:
* **Tratamento de Multicolinearidade:** As variáveis de atraso foram consolidadas em novas features (`total_atrasos_graves` e `ocorreu_atraso_grave`) para eliminar a redundância.
* **Criação de Pipeline:** Foi construído um `Pipeline` robusto para automatizar o pré-processamento, incluindo imputação de dados faltantes, escalonamento de features numéricas (`StandardScaler`) e codificação de variáveis categóricas (`OneHotEncoder`).

### 4. Modelagem Preditiva
Foram treinados e avaliados três modelos distintos para comparar suas performances:
1.  **Regressão Logística:** Utilizado como um baseline robusto, com ajuste para o desbalanceamento.
2.  **Random Forest:** Um modelo de ensemble baseado em árvores de decisão.
3.  **XGBoost:** Um modelo de Gradient Boosting, conhecido por sua alta performance.

---

## Resultados e Conclusão

O **XGBoost** se consolidou como o modelo campeão, apresentando o maior poder de discriminação com um **AUC-ROC de 0.85**.

![Curva ROC Comparativa](https://github.com/Sthefanie-neuro/analise_credito/blob/main/image_5af321.png)

A análise das métricas revelou um claro trade-off entre a detecção de risco (Recall) e a certeza da previsão (Precisão), levando à escolha do XGBoost por seu maior potencial preditivo latente, que pode ser calibrado conforme a estratégia de risco do negócio.

### Recomendação Final
O modelo **XGBoost** foi escolhido como a melhor solução. Recomenda-se sua implementação não como um sistema de decisão binário (aprova/nega), mas como uma **ferramenta de pontuação de risco**. Clientes com alta probabilidade de inadimplência devem ser encaminhados para uma análise de crédito mais detalhada.

---

## Como Executar o Projeto

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/Sthefanie-neuro/analise_credito.git](https://github.com/Sthefanie-neuro/analise_credito.git)
    ```
2.  **Crie um ambiente virtual e instale as dependências:**
    ```bash
    cd analise_credito
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Execute o script de carga de dados:**
    Este passo criará o banco de dados `analise_de_credito.db` necessário para o notebook.
    ```bash
    python 01_carga_sql.py
    ```
4.  **Execute o Jupyter Notebook:**
    Abra o arquivo `analise_credito.ipynb` em um ambiente Jupyter (como Jupyter Lab ou VS Code) e execute as células na ordem.

## Tecnologias Utilizadas
* Python
* Pandas & NumPy
* Scikit-learn
* XGBoost
* Matplotlib & Seaborn
* SQLite3
* Jupyter Notebook

---

## Contato

**Sthefanie Otaviano**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](http://linkedin.com/in/sthefanie-ferreira-de-s-d-otaviano-976a59206)
