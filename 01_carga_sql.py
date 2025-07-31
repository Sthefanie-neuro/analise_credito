import pandas as pd
import sqlite3

# --- Configuração ---
ARQUIVO_CSV = 'cs-training.csv'
NOME_BANCO_DE_DADOS = 'analise_de_credito.db'
NOME_DA_TABELA = 'clientes_credito'

print(f"Lendo o arquivo CSV: {ARQUIVO_CSV}")
# Carrega os dados do CSV para o pandas
df = pd.read_csv(ARQUIVO_CSV)

# Remove a primeira coluna 'Unnamed: 0' que é apenas um índice antigo
df = df.drop(columns=['Unnamed: 0'])

print("Conectando ao banco de dados...")
# Cria a conexão com o banco de dados SQLite
# Se o arquivo não existir, ele será criado
conn = sqlite3.connect(NOME_BANCO_DE_DADOS)

print(f"Salvando dados na tabela '{NOME_DA_TABELA}'...")
# Usa a função to_sql para salvar o dataframe na tabela especificada
# if_exists='replace' garante que, se rodarmos o script de novo, a tabela antiga será substituída
df.to_sql(NOME_DA_TABELA, conn, if_exists='replace', index=False)

print("Fechando conexão com o banco de dados.")
# Fecha a conexão
conn.close()

print("\n✅ Processo concluído! O banco de dados foi criado e populado com sucesso.")