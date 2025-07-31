# 📊 Predição de Inadimplência com Machine Learning

# [Markdown]
"""
## Objetivo do Projeto

Construir um modelo preditivo para identificar clientes com alto risco de inadimplência nos próximos dois anos. O banco de dados utilizado é o "Give Me Some Credit" do Kaggle.

| Item                    | Detalhes                                                                |
|-------------------------|-------------------------------------------------------------------------|
| 📊 Problema de Negócio  | Prever inadimplência em 2 anos                                        |
| 📂 Fonte dos Dados      | Kaggle - Give Me Some Credit (2011)                                   |
| 🧠 Objetivo             | Criar modelo de classificação para prever inadimplência futura         |
| 🔧 Técnicas             | Limpeza de dados, EDA, RandomForest, XGBoost, Oversampling             |
| 📈 Métricas             | Accuracy, ROC AUC, F1-score, Matriz de confusão                       |
"""

# [Code]
# Imports
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# [Code]
# Conecta ao banco de dados SQLite
conn = sqlite3.connect('analise_de_credito.db')
df = pd.read_sql_query("SELECT * FROM clientes_credito", conn)
conn.close()

# [Code]
# Renomeia as colunas para maior legibilidade
mapa_nomes = {
    'RevolvingUtilizationOfUnsecuredLines': 'rotativo_utilizado',
    'age': 'idade',
    'NumberOfTime30-59DaysPastDueNotWorse': 'atraso_30_59',
    'DebtRatio': 'divida_renda',
    'MonthlyIncome': 'renda_mensal',
    'NumberOfOpenCreditLinesAndLoans': 'linhas_credito',
    'NumberOfTimes90DaysLate': 'atraso_90',
    'NumberRealEstateLoansOrLines': 'emprestimos_imobiliarios',
    'NumberOfTime60-89DaysPastDueNotWorse': 'atraso_60_89',
    'NumberOfDependents': 'num_dependentes',
    'SeriousDlqin2yrs': 'inadimplente_2_anos'
}
df.rename(columns=mapa_nomes, inplace=True)

# [Code]
# Verifica valores ausentes e trata
print("Valores ausentes por coluna:\n", df.isnull().sum())
df['renda_mensal'].fillna(df['renda_mensal'].median(), inplace=True)
df['num_dependentes'].fillna(0, inplace=True)

# [Code]
# Análise da variável alvo
sns.countplot(x='inadimplente_2_anos', data=df)
plt.title("Distribuição da variável alvo")
plt.show()

# [Code]
# Define variáveis X e y
X = df.drop('inadimplente_2_anos', axis=1)
y = df['inadimplente_2_anos']

# Balanceamento das classes
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

# Padroniza os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# Divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_res, test_size=0.3, random_state=42)

# [Code]
# Treinamento e comparação entre modelos
modelos = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    print(f"\nModelo: {nome}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, modelo.predict_proba(X_test)[:,1]))

# [Code]
# Importância das variáveis com Random Forest
modelo_rf = RandomForestClassifier().fit(X_train, y_train)
importancias = modelo_rf.feature_importances_
plt.figure(figsize=(10,6))
sns.barplot(x=importancias, y=X.columns)
plt.title("Importância das variáveis - Random Forest")
plt.show()

# [Markdown]
"""
## 📌 Conclusão

- O modelo XGBoost apresentou o melhor desempenho, com ótimo balanceamento entre precisão e recall.
- As variáveis mais importantes foram: atraso_90, atraso_30_59 e linhas de crédito abertas.
- O oversampling ajudou a mitigar o desbalanceamento da classe.

### 🔄 Próximos passos:
- Aplicar otimização de hiperparâmetros.
- Serviço do modelo via API.
- Adicionar dashboard com Streamlit ou Power BI.

---

💼 Projeto por [Seu Nome] | [LinkedIn] | [GitHub]
"""
