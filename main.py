#Importando bibliotecas
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings

#Tornar o código mais limpo
warnings.filterwarnings("ignore", category=FutureWarning)

# Pega o caminho absoluto do diretório onde o script está
diretorio_script = os.path.dirname(__file__)

# Junta o caminho do diretório com a pasta 'dados' e o nome do arquivo
caminho_arquivo_apps = os.path.join(diretorio_script, 'dados', 'application_record.csv')
caminho_arquivo_credits = os.path.join(diretorio_script, 'dados', 'credit_record.csv')
print(f"Lendo os arquivos de: {caminho_arquivo_apps} e {caminho_arquivo_credits}")

# 1. Define o caminho para a pasta 'graficos'
# Path(__file__).parent pega o diretório do script atual
diretorio_graficos = Path(__file__).parent / 'graficos'

# 2. Cria a pasta 'graficos' se ela não existir
# O 'exist_ok=True' evita erros se a pasta já existir
diretorio_graficos.mkdir(exist_ok=True)

#Carregando os dados 

df_apps = pd.read_csv(caminho_arquivo_apps)

df_credits = pd.read_csv(caminho_arquivo_credits)

'''
Legenda da Situação

C: Pago no mês

X: Sem empréstimo no mês

0: Pago em dia

1: Atraso de 1-29 dias

2: Atraso de 30-59 dias

3: Atraso de 60-89 dias

4: Atraso de 90-119 dias

5: Atraso de 120-149 dias

6: Dívidas vencidas ou em atraso por mais de 150 dias

Considerando:

    mau pagador:se o cliente já teve algum mês com atraso significativo (status 2, 3, 4, 5),
ele será classificado como "mau pagador" (risco alto).

    "bom pagador": se o cliente teve um atraso inferior a 30 dias (status 1), vamos considerar isso um deslize
pontual e não sinal de alto risco ainda (risco moderado).

    bom pagador: se o cliente pagou durante o mês ou no mesmo dia,
ele será classificado como "bom pagador" (risco baixo).
'''
# Classificando como maus pagadores cliente com '2', '3', '4', '5' dentro da coluna STATUS
df_credits['mau_pagador'] = np.where(df_credits['STATUS'].isin(['2', '3', '4', '5']), 1, 0)

# Agora, para cada cliente (ID), se ele teve 'mau_pagador' == 1 em qualquer mês, ele é um mau pagador.
# Usamos groupby() e max() para isso.
risco_por_cliente = df_credits.groupby('ID')['mau_pagador'].max().reset_index()

# Renomear a coluna para ser nossa variável alvo
risco_por_cliente.rename(columns={'mau_pagador': 'alvo_risco'}, inplace=True)

# Juntando os dois dataframes pela coluna ID
df = pd.merge(df_apps, risco_por_cliente, on='ID', how='inner') # O how='inner' garante que só manteremos os clientes que estão em ambos os arquivos.

#Análise Exploratória e Pré-processamento 

# Verificar como a nossa nova variável alvo ficou distribuída
print(df['alvo_risco'].value_counts())
sns.countplot(x='alvo_risco', data=df)

''' Análise de Balanceamento da Variável Alvo
Ao analisar a nossa variável alvo_risco, observamos um forte desbalanceamento de classes.
Observamos que o resultado de 35.841 clientes "bons" (classe 0) para apenas 616 "maus" (classe 1)
mostra um desbalanceamento severo. Se ignorarmos isso, o modelo pode simplesmente aprender a chutar "bom pagador"
para todo mundo e ainda assim ter uma acurácia altíssima (mais de 98%), mesmo sendo completamente inútil para o negócio, que é justamente identificar os maus pagadores. 
 
Para mitigar esse problema e garantir que nosso modelo aprenda a identificar os maus pagadores, aplicaremos uma técnica de reamostragem.
A técnica escolhida será o SMOTE, que será aplicada apenas no conjunto de treinamento para evitar vazamento de dados.'''

# --- 1. Verificação de valores não nulos ---
qtd_nulos = df.isnull().sum()
porcentagem_nulos = (df.isnull().sum() / len(df)) * 100
null_df = pd.DataFrame({'Total_Nulos': qtd_nulos, 'Porcentagem_Nulos': porcentagem_nulos})
# Mostra apenas as colunas que de fato têm valores nulos
print(null_df[null_df['Total_Nulos'] > 0])

# --- 2. Tratamento de Nulos e Remoção de Colunas ---

df_processado = df.copy()

# Remover a coluna 'OCCUPATION_TYPE' por ter alta porcentagem de nulos
if 'OCCUPATION_TYPE' in df_processado.columns:
    print("\nRemovendo a coluna 'OCCUPATION_TYPE'...")
    df_processado.drop('OCCUPATION_TYPE', axis=1, inplace=True)

# Remover colunas irrelevantes.
print("Removendo a coluna 'ID' por não ser uma feature preditiva.")
df_processado.drop('ID', axis=1, inplace=True)

# --- 3. Transformação de Colunas Categóricas ---
print("\n--- Transformando Colunas Categóricas em Numéricas ---")
colunas_categoricas = df_processado.select_dtypes(include=['object']).columns
print(f"Colunas a serem transformadas: {list(colunas_categoricas)}")

df_processado = pd.get_dummies(df_processado, columns=colunas_categoricas, drop_first=True)
print(f"O número total de colunas agora é: {df_processado.shape[1]}")


# --- 4. Preparação Final para o Modelo ---
    
# Separando as features (X) da variável alvo (y)
X = df_processado.drop('alvo_risco', axis=1)
y = df_processado['alvo_risco']
    
print("\n--- Pré-processamento Concluído ---")
print("As variáveis 'X' (features) e 'y' (alvo) estão prontas.")
print("O próximo passo é usar X e y para o train_test_split e, em seguida, aplicar o SMOTE no conjunto de treino.")
    
# Exibindo as primeiras linhas e colunas do X para verificação
print("\nCabeçalho das features (X) após o pré-processamento:")
print(X.head())

# --- 5. Análise Expolatória dos dados ---
print("\nSelecionando e preparando features para o heatmap...")

# Selecionar um subconjunto de colunas para um gráfico mais legível
features_selecionadas = [
    'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
    'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS', 'alvo_risco'
]
df_heatmap = df[features_selecionadas].copy()

# Fazer transformações para deixar os dados mais intuitivos
df_heatmap['CODE_GENDER'] = df_heatmap['CODE_GENDER'].replace({'M': 0, 'F': 1})
df_heatmap['FLAG_OWN_CAR'] = df_heatmap['FLAG_OWN_CAR'].replace({'N': 0, 'Y': 1})
df_heatmap['FLAG_OWN_REALTY'] = df_heatmap['FLAG_OWN_REALTY'].replace({'N': 0, 'Y': 1})

# Converter dias em anos (e deixar positivo)
df_heatmap['IDADE_ANOS'] = -df_heatmap['DAYS_BIRTH'] / 365
# O número de dias de emprego é negativo. Positivo significa 'desempregado'.
# Vamos converter os negativos para positivos e zerar os desempregados para a correlação.
df_heatmap['ANOS_EMPREGADO'] = -df_heatmap['DAYS_EMPLOYED'] / 365
df_heatmap.loc[df_heatmap['ANOS_EMPREGADO'] < 0, 'ANOS_EMPREGADO'] = 0

# Remover colunas originais de "dias"
df_heatmap.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)

print("Features selecionadas e transformadas.")


# --- 6. Análise Expolatória dos dados (Calcular e Plotar o Heatmap) ---
print("Gerando o heatmap de correlação...")

# Calcular a matriz de correlação
corr_matrix = df_heatmap.corr()

# Criar uma máscara para mostrar apenas a parte triangular inferior do gráfico
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Configurar a figura do Matplotlib
f, ax = plt.subplots(figsize=(12, 10))

# Gerar o heatmap
sns.heatmap(corr_matrix, 
            mask=mask, 
            annot=True,     # Adiciona os números em cada célula
            fmt='.2f',      # Formata os números para 2 casas decimais
            cmap='vlag',    # Paleta de cores (vermelho-azul)
            linewidths=.5,  # Linhas entre as células
            cbar_kws={"shrink": .75}) # Ajusta o tamanho da barra de cores

ax.set_title('Heatmap de Correlação entre Features', fontsize=18)
plt.xticks(rotation=45, ha='right') # Rotaciona os nomes das colunas para não sobrepor
plt.tight_layout()

caminho_heatmap = diretorio_graficos / 'heatmap_correlacao.png'
# Salvar a imagem em um arquivo
plt.savefig(caminho_heatmap)

print("\nGráfico 'heatmap_correlacao.png' foi salvo no seu diretório.")
print("Este gráfico mostra a correlação entre cada par de variáveis.")

# --- 7.  SMOTE e Regressão Logística. ---
# --- Etapa 1: Divisão em Treino e Teste ---
print("\n--- Etapa 1: Dividindo dados em treino e teste... ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,     # 30% para teste
    random_state=42,   # Para reprodutibilidade
    stratify=y         # ESSENCIAL para dados desbalanceados
)
print(f"Tamanho do treino: {X_train.shape[0]} amostras")
print(f"Tamanho do teste: {X_test.shape[0]} amostras")


# --- Etapa 2: Balanceamento com SMOTE (Apenas no Treino) ---
print("\n--- Etapa 2: Aplicando SMOTE nos dados de treino... ---")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("Distribuição de classes no treino após SMOTE:")
print(y_train_smote.value_counts())


# --- Etapa 3: Escalonamento de Features (StandardScaler) ---
# Modelos como Regressão Logística funcionam melhor com dados escalonados
print("\n--- Etapa 3: Escalonando as features... ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test) # Usar o mesmo scaler treinado no treino


# --- Etapa 4: Treinamento do Modelo ---
print("\n--- Etapa 4: Treinando o modelo de Regressão Logística... ---")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train_smote)
print("Modelo treinado com sucesso!")


# --- Etapa 5: Avaliação do Modelo ---
print("\n--- Etapa 5: Avaliando o modelo nos dados de teste... ---")
y_pred = model.predict(X_test_scaled)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Bom Pagador (0)', 'Mau Pagador (1)']))

# AUC Score
auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
print(f"AUC Score: {auc:.4f}")

# Matriz de Confusão
print("\nMatriz de Confusão:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Previsto Bom', 'Previsto Mau'], 
            yticklabels=['Real Bom', 'Real Mau'])
plt.title('Matriz de Confusão')
plt.ylabel('Verdadeiro')
plt.xlabel('Previsto')
caminho_matriz = diretorio_graficos / 'matriz_confusao_final.png'
plt.savefig(caminho_matriz)
plt.show()
print("Gráfico 'matriz_confusao_final.png' foi salvo.")
