# 🎬 YouTube Analytics - Modelagem Preditiva

Análise completa de Machine Learning com regressão e classificação em dados do YouTube.

**Status:** ✅ **COMPLETO E PRONTO PARA PRODUÇÃO**

---

## 🎯 Objetivos

✅ Exploração e análise exploratória de dados (EDA)  
✅ Testes estatísticos (Correlação, T-test, ANOVA, Chi²)  
✅ Modelagem preditiva (Regressão + Classificação)  
✅ Otimização de hiperparâmetros (Grid Search + Random Search)  
✅ Diagnóstico e interpretabilidade de modelos  
✅ Deploy com funções reutilizáveis  

---

## 📊 Funcionalidades

| Funcionalidade | Descrição |
|---|---|
| 📈 **EDA** | Histogramas, boxplots, heatmaps, outliers (IQR) |
| 🔬 **Testes Estatísticos** | Pearson, Spearman, T-test, ANOVA, Chi² |
| 📉 **Regressão** | Linear + Polinomial (grau 2) |
| 🎲 **Classificação** | Naive Bayes + Logistic Regression |
| ⚙️ **Otimização** | Grid Search (4 comb.) + Random Search (20 iter.) |
| ✅ **Validação** | K-Fold (5 splits) + Diagnóstico de resíduos |
| 📊 **Interpretação** | Coeficientes, Odds Ratio, Feature Importance |
| 💾 **Deploy** | Salvar modelos (.pkl) + Funções de predição |

---

## 🛠️ Tecnologias

| Biblioteca | Uso |
|---|---|
| **Python 3.x** | Linguagem |
| **Pandas** | Manipulação de dados |
| **NumPy** | Computação numérica |
| **Scikit-learn** | Machine Learning |
| **Statsmodels** | Análise estatística |
| **SciPy** | Testes estatísticos |
| **Matplotlib & Seaborn** | Visualização |
| **Jupyter** | Notebook interativo |

---

## 📁 Estrutura

```
YouTube-Analytics/
├── README.md                           # Este arquivo
├── notebook.ipynb                      # Análise completa
├── youtube_recommendation_dataset.csv  # Dataset
└── models/                             # Gerado automaticamente
    ├── grid_search_model.pkl           # Melhor modelo
    ├── scaler.pkl                      # Padronizador
    ├── feature_names.pkl               # Nomes das features
    └── metadata.pkl                    # Metadados
```

---

## 🚀 Como Executar

### Pré-requisitos

```bash
pip install pandas numpy scikit-learn statsmodels scipy matplotlib seaborn jupyter
```

### Passos

1. **Abra o notebook**
   ```bash
   jupyter notebook notebook.ipynb
   ```
   Ou abra com VS Code (com extensão Jupyter)

2. **Execute as células na sequência** (Shift + Enter)

3. **Analise os resultados** gerados automaticamente

---

## 📈 Pipeline Resumido

```
Dataset → EDA → Testes Estatísticos → Preparação
    ↓
Regressão (Linear + Polinomial) → Classificação (Naive Bayes + Logistic)
    ↓
Validação (K-Fold) → Otimização (Grid/Random Search)
    ↓
Diagnóstico (Resíduos, VIF) → Interpretação (Coef., Feature Importance)
    ↓
Deploy (Salvar modelos + Funções prontas)
```

---

## 📊 Métricas

**Regressão:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

**Classificação:**
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC (classificação binária)
- Matriz de Confusão

---

## 🔍 Variável-Alvo (Automático)

**Regressão:** `views`, `view_count`, `likes`, `comment_count`, `watch_time`  
**Classificação:** `category`, `label`, `genre`, `is_recommended`

---

## ⚙️ Parâmetros

```python
RANDOM_SEED = 42    # Reprodutibilidade
TEST_SIZE = 0.3     # 30% teste + validação
VAL_SIZE = 0.5      # 50% do teste para validação
N_SPLITS = 5        # K-Fold com 5 splits
POLY_DEGREE = 2     # Grau do polinômio
MAX_ITER = 1000     # Iterações máximas
```

---

## 📝 Resultados Esperados

✅ Tabelas comparativas (MAE, RMSE, R², F1)  
✅ 8+ gráficos (distribuições, correlações, confusão)  
✅ Testes estatísticos com p-values  
✅ Coeficientes padronizados e não-padronizados  
✅ Feature importance relativa (%)  
✅ Diagnóstico: normalidade, homocedasticidade, multicolinearidade  
✅ Curvas ROC-AUC (classificação)  

---

## 🚀 Deploy em Produção

### Regressão
```python
from joblib import load

model = load('models/grid_search_model.pkl')
predictions = model.predict(novo_dados)
print(f"Visualizações preditas: {predictions[0]:.0f}")
```

### Classificação
```python
from joblib import load

model = load('models/grid_search_clf.pkl')
scaler = load('models/scaler_clf.pkl')

X_scaled = scaler.transform(novo_dados)
classe = model.predict(X_scaled)[0]
confianca = model.predict_proba(X_scaled).max()
print(f"Categoria: {classe} | Confiança: {confianca:.2%}")
```
---

## 🎓 Próximos Passos

- [ ] Feature Engineering avançado
- [ ] Algoritmos: Random Forest, XGBoost, Neural Networks
- [ ] Tratamento de desbalanceamento (SMOTE)
- [ ] API REST (Flask/FastAPI)
- [ ] Dashboard (Streamlit/Dash)

---

## 📚 Referências

- [Scikit-learn](https://scikit-learn.org/)
- [Statsmodels](https://www.statsmodels.org/)
- [Pandas](https://pandas.pydata.org/)
- [Seaborn](https://seaborn.pydata.org/)

---

## 📋 Checklist Completo

- ✅ EDA (8+ visualizações)
- ✅ Testes estatísticos (Pearson, T-test, ANOVA, Chi²)
- ✅ Regressão Linear + Polinomial
- ✅ Classificação (Naive Bayes + Logistic)
- ✅ K-Fold Cross-Validation
- ✅ Grid Search + Random Search
- ✅ Diagnóstico de resíduos
- ✅ Interpretação de coeficientes
- ✅ Feature Importance
- ✅ Matriz de Confusão + ROC-AUC
- ✅ Salvar modelos (.pkl)
- ✅ Funções prontas de predição

---


