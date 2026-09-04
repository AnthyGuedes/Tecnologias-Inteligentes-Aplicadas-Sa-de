# Relatório de Avaliação dos Modelos de Machine Learning

Este documento consolida os resultados obtidos na execução dos scripts de classificação para os dois problemas avaliados no projeto: **Predição de Compra de Produto** ([Renda.py](file:///c:/TecnologiaAplicadaSaude/Renda.py)) e **Predição de Risco de Internação** ([Saude.py](file:///c:/TecnologiaAplicadaSaude/Saude.py)).

---

## 1. Problema 1: Predição de Compra de Produto (`Renda.py`)

* **Base de Dados:** `dados_predicao_modelos.csv`
* **Variáveis Preditivas (Features):** `Idade`, `Renda_Anual_K`, `Score_Credito`, `Pontuacao_Engajamento`
* **Alvo (Target):** `Compro_Produto`
* **Estratégia de Validação:** Divisão 70% treino e 30% teste com estratificação (`stratify=y`, `random_state=42`)
* **Pré-processamento:** Padronização com `StandardScaler`

### Ranking Final (Problema 1)

| Posição | Modelo | Acurácia | F1-Score (Macro) |
| :---: | :--- | :---: | :---: |
| **1º** | **Random Forest** | **0.6933** | **0.6514** |
| 2º | Decision Tree | 0.6533 | 0.6346 |
| 3º | KNN | 0.6533 | 0.6238 |
| 4º | SVM | 0.6800 | 0.6237 |
| 5º | Logistic Regression | 0.6533 | 0.6100 |
| 6º | Naive Bayes | 0.6533 | 0.6100 |
| 7º | Gradient Boosting | 0.6000 | 0.5297 |

### Detalhamento por Modelo (Problema 1)

#### 1. Logistic Regression
* **Acurácia:** 0.6533
* **F1-Score (Macro):** 0.6100
* **Matriz de Confusão:**
  ```text
  [[37  8]
   [18 12]]
  ```

#### 2. Decision Tree
* **Acurácia:** 0.6533
* **F1-Score (Macro):** 0.6346
* **Matriz de Confusão:**
  ```text
  [[33 12]
   [14 16]]
  ```

#### 3. Random Forest (Melhor Desempenho)
* **Acurácia:** 0.6933
* **F1-Score (Macro):** 0.6514
* **Matriz de Confusão:**
  ```text
  [[39  6]
   [17 13]]
  ```

#### 4. KNN (K-Nearest Neighbors)
* **Acurácia:** 0.6533
* **F1-Score (Macro):** 0.6238
* **Matriz de Confusão:**
  ```text
  [[35 10]
   [16 14]]
  ```

#### 5. Naive Bayes
* **Acurácia:** 0.6533
* **F1-Score (Macro):** 0.6100
* **Matriz de Confusão:**
  ```text
  [[37  8]
   [18 12]]
  ```

#### 6. SVM (Support Vector Machine)
* **Acurácia:** 0.6800
* **F1-Score (Macro):** 0.6237
* **Matriz de Confusão:**
  ```text
  [[40  5]
   [19 11]]
  ```

#### 7. Gradient Boosting
* **Acurácia:** 0.6000
* **F1-Score (Macro):** 0.5297
* **Matriz de Confusão:**
  ```text
  [[37  8]
   [22  8]]
  ```

---

## 2. Problema 2: Predição de Risco de Internação (`Saude.py`)

* **Base de Dados:** `dados_saude_predicao.csv`
* **Variáveis Preditivas (Features):** `Idade`, `Pressao_Arterial`, `Colesterol_Total`, `Frequencia_Cardiaca_Max`
* **Alvo (Target):** `Risco_Internacao`
* **Estratégia de Validação:** Divisão 70% treino e 30% teste com estratificação (`stratify=y`, `random_state=42`)
* **Pré-processamento:** Padronização com `StandardScaler`

### Ranking Final (Problema 2)

| Posição | Modelo | Acurácia | F1-Score (Macro) |
| :---: | :--- | :---: | :---: |
| **1º** | **Naive Bayes** | **0.8133** | **0.8125** |
| 2º | Gradient Boosting | 0.7867 | 0.7863 |
| 3º | Logistic Regression | 0.7600 | 0.7600 |
| 4º | Decision Tree | 0.7600 | 0.7596 |
| 5º | SVM | 0.7467 | 0.7467 |
| 6º | Random Forest | 0.7467 | 0.7459 |
| 7º | KNN | 0.6933 | 0.6925 |

### Detalhamento por Modelo (Problema 2)

#### 1. Logistic Regression
* **Acurácia:** 0.7600
* **F1-Score (Macro):** 0.7600
* **Matriz de Confusão:**
  ```text
  [[28 10]
   [ 8 29]]
  ```

#### 2. Decision Tree
* **Acurácia:** 0.7600
* **F1-Score (Macro):** 0.7596
* **Matriz de Confusão:**
  ```text
  [[27 11]
   [ 7 30]]
  ```

#### 3. Random Forest
* **Acurácia:** 0.7467
* **F1-Score (Macro):** 0.7459
* **Matriz de Confusão:**
  ```text
  [[30  8]
   [11 26]]
  ```

#### 4. KNN (K-Nearest Neighbors)
* **Acurácia:** 0.6933
* **F1-Score (Macro):** 0.6925
* **Matriz de Confusão:**
  ```text
  [[24 14]
   [ 9 28]]
  ```

#### 5. Naive Bayes (Melhor Desempenho)
* **Acurácia:** 0.8133
* **F1-Score (Macro):** 0.8125
* **Matriz de Confusão:**
  ```text
  [[33  5]
   [ 9 28]]
  ```

#### 6. SVM (Support Vector Machine)
* **Acurácia:** 0.7467
* **F1-Score (Macro):** 0.7467
* **Matriz de Confusão:**
  ```text
  [[28 10]
   [ 9 28]]
  ```

#### 7. Gradient Boosting
* **Acurácia:** 0.7867
* **F1-Score (Macro):** 0.7863
* **Matriz de Confusão:**
  ```text
  [[31  7]
   [ 9 28]]
  ```

---

## 3. Síntese Comparativa

* **Problema 1 (Compra - Renda.py):**
  * O modelo **Random Forest** obteve o melhor equilíbrio geral, alcançando **69,33% de acurácia** e **F1-Score Macro de 0,6514**.
* **Problema 2 (Saúde / Internação - Saude.py):**
  * O modelo **Naive Bayes** destacou-se com a melhor performance, atingindo **81,33% de acurácia** e **F1-Score Macro de 0,8125**, seguido de perto pelo **Gradient Boosting** (**78,67%** de acurácia).
