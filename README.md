### TP3 — Análisis de Sentimiento sobre Tweets (Sentiment140)



#### Diplomatura IA - Universidad de Palermo



**Alumno: Gonzalez Marta Elizabeth**

**Profesor: Ing. Jaureguy Martin**

**Mes: Julio26**

**Trabajo Practico 3**

\---



### Estructura del proyecto

```
.TP3
├── README.md
├── requirements.txt
├── data/raw/
│   ├── testdata\_manual\_2009\_06\_14.csv
│   ├── training.1600000.processed.noemoticon.csv   

├── data/processed/
│   ├── mega\_clean.csv                    # (ref) Notebook 1 -> une datasets + limpia + feature engineering
│   ├── split\_train.csv / split\_val.csv / split\_test.csv   # (ref) Notebook 2 -> split 70/15/15
│   ├── tfidf\_vectorizer\_mega.pkl, scaler\_features.pkl      # (ref) Notebook 2
│   ├── nb\_model.pkl, lr\_model\_mega.pkl, lr\_fe\_model.pkl    # (ref) Notebook 2 -> 3 modelos candidatos
│   ├── modelo\_final\_elegido.pkl          # (ref) Notebook 3 -> el modelo elegido tras validación
│   ├── comparacion\_validacion.csv, tabla\_generalizacion\_final.csv, comparacion\_final\_vs\_textblob.csv  # (ref) Notebook 3
│   └── predicciones\_test.csv             # (ref) Notebook 4
└── notebooks/
    ├── 00\_primera\_lectura\_eda.ipynb
    ├── 01\_union\_eda\_preprocesamiento.ipynb
    ├── 02\_modelado\_sentimiento.ipynb
    ├── 03\_analisis\_errores.ipynb
    ├── 04\_predicciones.ipynb
    └── 05\_topicos\_embeddings\_grafos.ipynb
```

Correr en orden 0 → 1 → 2 → 3 → 4 → 5 (el notebook 0 es independiente, no genera archivos que usen los demás).

\---



### Instalación

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

Descargas automáticas la primera vez: NLTK (`stopwords`, `wordnet`, `omw-1.4`, `opinion\_lexicon`) en el notebook 1, y GloVe-Twitter-25 (\~105 MB) en el notebook 5.

### 

\---

### Qué hace cada notebook?



#### 00\_primera\_lectura\_eda.ipynb 



Primera lectura de los 2 datasets crudos, por separado, sin ninguna unión ni limpieza. Nulos, duplicados, boxplots de outliers de longitud (por IQR), longitud por clase, top usuarios más activos, volumen de tweets por día, correlación inicial. No genera ni consume archivos de los demás notebooks.



#### 01\_union\_eda\_preprocesamiento.ipynb



Une 1.6M + test manual (con la clase neutral incluida, documentando su desbalance), EDA (temporalidad, correlación de atributos), limpieza de texto, y 

**Feature Engineering** (14 features: puntuación, mayúsculas, elongaciones, emoticones, menciones/hashtags, score de polaridad léxica). 

Guarda `TP3/data/proceseed/mega\_clean.csv`.



#### 02\_modelado\_sentimiento.ipynb 



* **Split train/test (70/30, estratificado)** — sin conjunto de validación.
* Entrena 3 modelos candidatos: Naive Bayes, Regresión Logística (solo TF-IDF), Regresión Logística + Feature Engineering.
* **Los 4 modelos (incluido TextBlob) se evalúan con `classification\_report` en TRAIN y TEST**, impresos secuencialmente para cada uno.
* Tabla y gráfico comparativo como síntesis, matrices de confusión en test.
* Selección del mejor modelo según F1 macro en test.
* Guarda los 3 modelos + el modelo elegido.



#### 03\_analisis\_errores.ipynb 



* Toma el modelo ya elegido en el Notebook 2 y analiza sus errores en profundidad.
* Similitud coseno entre tweets de test mal clasificados y sus vecinos más parecidos en train.
* Distribución de errores por clase real (confirma que la clase neutral concentra la mayor proporción de errores).

#### 

#### 04\_predicciones.ipynb



* Carga el modelo ya elegido y validado.
* Genera predicciones detalladas sobre test (texto + etiqueta real + predicha), con ejemplos de aciertos y errores.
* Función reutilizable `predecir\_sentimiento(texto)` para clasificar texto nuevo (aplicando la misma limpieza del Notebook 1).
* Exporta `TP3/data/processed/predicciones\_test.csv`.

#### 

#### 05\_topicos\_embeddings\_grafos.ipynb (más tópicos + búsqueda de eventos)



* BERTopic ajustado en muestra (subida de 8% a 20%) + propagado al resto (`transform()`), con `PCA+KMeans` (subido de 10 a 40 clusters) para mayor granularidad de tópicos.
* **Búsqueda dirigida** de eventos puntuales de la época (muerte de Michael Jackson/Farrah Fawcett, elección de Irán, Economia) tanto en las keywords de los tópicos como directamente en el texto, con su ubicación temporal — para verificar si el dataset capturó esos eventos reales.
* Word2Vec propio entrenado sobre el megaset completo (>1.6M tweets), comparado contra GloVe-Twitter pre-entrenado.
* PMI de bigramas por clase.
* Grafo de menciones entre usuarios (con centralidad y comunidades) y grafo de co-ocurrencia de hashtags, ambos con su evolución temporal semana a semana.

\---

## 

## 

### Decisiones y limitaciones clave (resumen)

* La clase neutral queda en fuerte desbalance (139 de >1.6M) — documentado explícitamente; los modelos usan balanceo de clases.
* La separación entrenar (Notebook 2) → validar/elegir (Notebook 3) → predecir (Notebook 4) evita "espiar" el test durante el desarrollo, y hace que la métrica final de test sea una medición honesta de generalización.
* BERTopic no se ajusta sobre el megaset completo por razones de escala — se usa la estrategia estándar de muestra + transform.

