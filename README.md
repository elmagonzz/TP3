## TP3 — Análisis de Sentimiento sobre Tweets (Sentiment140)



1. **Split de 3 partes** (train/validación/test, 70/15/15) en vez de solo train/test.
2. Cada modelo se evalúa con `classification\\\_report` **en train Y en validación** (no solo test), para poder detectar overfitting.
3. **Notebook dedicado a validación y selección del mejor modelo**: se comparan los candidatos en validación, se elige el mejor, y **recién ahí** se hace la única evaluación final sobre test (que hasta ese momento no fue tocado).
4. **Notebook dedicado a predicciones**: uso del modelo ya elegido para predecir sobre test y sobre texto nuevo.

\---





## Estructura del proyecto

```
.
├── README.md
├── requirements.txt
├── ../data/raw/
│   ├── testdata\\\_manual\\\_2009\\\_06\\\_14.csv

│   ├──training.1600000.processed.noemoticon  # dataset 1.6M

├── ../data/processed/ 

│   ├── mega\\\_clean.csv                    # (ref) Notebook 1 -> une datasets + limpia + feature engineering
│   ├── split\\\_train.csv / split\\\_val.csv / split\\\_test.csv   # (ref) Notebook 2 -> split 70/15/15
│   ├── tfidf\\\_vectorizer\\\_mega.pkl, scaler\\\_features.pkl      # (ref) Notebook 2
│   ├── nb\\\_model.pkl, lr\\\_model\\\_mega.pkl, lr\\\_fe\\\_model.pkl    # (ref) Notebook 2 -> 3 modelos candidatos
│   ├── modelo\\\_final\\\_elegido.pkl          # (ref) Notebook 3 -> el modelo elegido tras validación
│   ├── comparacion\\\_validacion.csv, tabla\\\_generalizacion\\\_final.csv, comparacion\\\_final\\\_vs\\\_textblob.csv  # (ref) Notebook 3
│   └── predicciones\\\_test.csv             # (ref) Notebook 4
└── notebooks/

\&#x20;   ├── 00\\\_primera\\\_lectura\\\_eda.ipynb
    ├── 01\\\_union\\\_eda\\\_preprocesamiento.ipynb
    ├── 02\\\_modelado\\\_sentimiento.ipynb
    ├── 03\\\_validacion\\\_seleccion\\\_modelo.ipynb
    ├── 04\\\_predicciones.ipynb
    └── 05\\\_topicos\\\_embeddings\\\_grafos.ipynb
```

Correr en orden 0 -> 1 → 2 → 3 → 4 → 5.

\---



## 

### Instalación

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\\\\Scripts\\\\activate
pip install -r requirements.txt
```

Descargas automáticas la primera vez: NLTK (`stopwords`, `wordnet`, `omw-1.4`, `opinion\\\_lexicon`) en el notebook 1, y GloVe-Twitter-25 (\~105 MB) en el notebook 5.

### 

\---

### Qué hace cada notebook

### 

### 00\_primera\_lectura\_eda.ipynb



Primer mirada cruda a ambos dataset para identificar nulos, duplicados, outliers, y primeras relaciones entre variables. No genera ningún archivo.



### 01\_union\_eda\_preprocesamiento.ipynb



Une 1.6M + test manual (con la clase neutral incluida, documentando su desbalance), EDA (temporalidad, correlación de atributos), limpieza de texto, y **Feature Engineering** (14 features: puntuación, mayúsculas, elongaciones, emoticones, menciones/hashtags, score de polaridad léxica). Guarda `../data/processed/mega\\\_clean.csv`.

### 

### 02\_modelado\_sentimiento.ipynb



* **Split de 3 partes** (train/validación/test, 70/15/15 estratificado) — guarda los 3 splits para reutilizarlos en los notebooks siguientes sin volver a splitear.
* Entrena 3 modelos candidatos: Naive Bayes, Regresión Logística (solo TF-IDF), Regresión Logística + Feature Engineering.
* **Cada modelo se evalúa con `classification\\\_report` en TRAIN y en VALIDACIÓN** — se compara el F1 de ambos para detectar overfitting (gap grande = mal síntoma).
* El **test no se toca en este notebook** — queda reservado para el Notebook 3.
* Guarda los 3 modelos entrenados + vectorizador + scaler.

### 

### 03\_validacion\_seleccion\_modelo.ipynb



* Compara los 3 candidatos **en validación** (F1 macro, F1 neutral).
* **Elige el mejor modelo** según ese criterio — no según el test, para no sesgar la elección.
* Hace la evaluación final — **una única vez** — sobre test, para el modelo elegido.
* Compara el modelo elegido contra TextBlob (pre-entrenado) sobre ese mismo test — la comparación "modelo entrenado vs. pre-entrenado" de la consigna original.
* Similitud coseno para interpretar errores del modelo final.
* Guarda el modelo elegido (`../data/processed/modelo\_final\_elegido.pkl`).



### 04\_predicciones.ipynb



* Carga el modelo ya elegido y validado.
* Genera predicciones detalladas sobre test (texto + etiqueta real + predicha), con ejemplos de aciertos y errores.
* Función reutilizable `predecir\\\_sentimiento(texto)` para clasificar texto nuevo (aplicando la misma limpieza del Notebook 1).
* Exporta `../data/processed/predicciones/\_test.csv`.

### 

### 05\_topicos\_embeddings\_grafos.ipynb



BERTopic a escala (muestra + `transform()`), Word2Vec propio vs. GloVe pre-entrenado, PMI, y los grafos de usuarios (menciones + hashtags) con su evolución temporal semana a semana.





\---



### Decisiones y limitaciones clave (resumen)



* La clase neutral queda en fuerte desbalance (139 de >1.6M) — documentado explícitamente; los modelos usan balanceo de clases.
* La separación entrenar (Notebook 2) → validar/elegir (Notebook 3) → predecir (Notebook 4) evita "espiar" el test durante el desarrollo, y hace que la métrica final de test sea una medición honesta de generalización.
* BERTopic no se ajusta sobre el megaset completo por razones de escala — se usa la estrategia estándar de muestra + transform.

