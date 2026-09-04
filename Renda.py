import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Modelos de classificação
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# 1. Carregar os dados
df = pd.read_csv('dados_predicao_modelos.csv')

# 2 e 3. Variáveis de interesse e remoção de nulos
features = ['Idade', 'Renda_Anual_K', 'Score_Credito', 'Pontuacao_Engajamento']
target = 'Compro_Produto'
df = df[features + [target]].dropna()

# 4. Features (X) e Target (y)
X = df[features]
y = df[target]

# 5. Divisão treino/teste com estratificação (70% treino / 30% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 6. Padronização dos atributos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Modelos do comparativo oficial
modelos = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# 8. Avaliação
resultados = []
print("=== AVALIAÇÃO: PROBLEMA 1 (COMPRA) ===\n")

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    resultados.append((nome, acc, f1))

    print(f"Modelo: {nome}")
    print(f"Acurácia: {acc:.4f} | F1-Score (Macro): {f1:.4f}")
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 50)

# 9. Ranking final
resultados.sort(key=lambda x: x[2], reverse=True)
print("\nRanking Final (Problema 1):")
print(f"{'Modelo':<25} {'Acurácia':<10} {'F1-Score (Macro)':<15}")
print("-" * 50)
for nome, acc, f1 in resultados:
    print(f"{nome:<25} {acc:<10.4f} {f1:<15.4f}")