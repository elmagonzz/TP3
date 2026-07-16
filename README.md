# TP3 — Análisis de Sentimiento sobre Tweets (Sentiment140)





##### Diplomatura en Inteligencia Artificial - UP

##### Alumno: Gonzalez Marta Elizabeth

##### Mes: Julio26

##### Trabajo Práctico 3





Análisis de sentimiento sobre tweets en inglés (dataset **Sentiment140**), combinando modelos clásicos de ML, un modelo pre-entrenado, modelado de tópicos con BERTopic, embeddings de palabras, análisis de grafos, métricas de asociación (coseno y PMI) y visualizaciones (wordclouds, UMAP, dendrogramas, redes).

\---



## 1\. Estructura del proyecto

```
.
├── README.md
├── requirements.txt
├── data/raw/
│   ├── testdata\_manual\_2009\_06\_14.csv       		# test manual oficial de Sentiment140 (3 clases)
│   ├── training.1600000.processed.noemoticon.csv     	# dataset original completo 1.6M registros de Tweets Sentiment140

├── data/processed/
│   ├── train\_clean.csv                      		# train limpio + features derivadas (generado por notebook 1)
│   ├── test\_clean.csv                       		# test limpio, subconjunto binario (generado por notebook 1)
│   ├── test\_full\_with\_neutral.csv           		# test completo con clase neutral (generado por notebook 1)
│   ├── tfidf\_vectorizer.pkl                 		# vectorizador TF-IDF ya ajustado (generado por notebook 2)
│   ├── lr\_model.pkl                         		# modelo de Regresión Logística ya entrenado (generado por notebook 2)
│   └── resultados\_modelos.csv               		# tabla comparativa de accuracy/F1 (generado por notebook 2)
└── notebooks/
    ├── 01\_EDA\_preprocesamiento.ipynb
    ├── 02\_modelado\_sentimiento.ipynb
    └── 03\_topicos\_embeddings\_visualizaciones.ipynb
```

Los notebooks están pensados para correrse **en orden** (1 → 2 → 3): cada uno lee archivos generados por el anterior. 

El notebook 1 ya genera todos los CSV intermedios, así que si solo se quiere revisar el modelado o los tópicos, alcanza con tener la carpeta `data/` tal como está en este entregable (ya incluye los archivos intermedios generados).

\---

## 

## 2\. Instalación



```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

### 

### Descargas adicionales la primera vez que se ejecutan los notebooks



* **NLTK**: el notebook 1 descarga automáticamente `stopwords`, `wordnet` y `omw-1.4` la primera vez (`nltk.download(...)`, ya incluido en el código, no requiere paso manual).
* **GloVe-Twitter (25 dim)**: el notebook 3 descarga automáticamente (vía `gensim.downloader`) un modelo de embeddings pre-entrenado en 2 mil millones de tweets (\~105 MB, se cachea localmente tras la primera descarga en `\~/gensim-data/`). Requiere conexión a internet la primera vez que se corre ese notebook.

\---

## 

## 3\. Datasets utilizados

|Archivo|Filas|Clases|Rol|
|-|-|-|-|
|`training.1600000.processed.noemoticon.csv`|1600000|0 (negativo) / 4 (positivo)|**Entrenamiento** — dataset original de 1.6M tweets|
|`testdata\_manual\_2009\_06\_14.csv`|498|0 / 2 (neutral) / 4|**Test externo** — set manualmente etiquetado por Stanford, con tópicos reales de Twitter 2009|



\---

## 

## 4\. Qué hace cada notebook

### 

### &#x09;01\_EDA\_preprocesamiento.ipynb



* Carga y comparación de los 3 datasets (nulos, duplicados, tipos).
* Distribución de clases, longitud de tweets (caracteres/palabras).
* **Análisis temporal**: parseo de la columna `date`, patrones de volumen y sentimiento por hora del día y día de la semana.
* **Relación entre atributos**: matriz de correlación entre longitud, momento temporal y sentimiento.
* Tópicos del test manual (columna `flag`).
* Limpieza de texto: decodificación de entidades HTML, eliminación de URLs/menciones, tratamiento de hashtags, remoción de stopwords conservando negaciones, lematización.
* WordClouds y top palabras por clase.
* Guarda los datasets limpios (`train\_clean.csv`, `test\_clean.csv`, `test\_full\_with\_neutral.csv`) para los notebooks siguientes.

### 

### &#x09;02\_modelado\_sentimiento.ipynb



* Vectorización TF-IDF (uni + bigramas).
* Entrena **2 modelos clásicos**: Naive Bayes Multinomial y Regresión Logística.
* Evalúa ambos, más **TextBlob** (modelo pre-entrenado), sobre el mismo test externo.
* Analiza por separado el subconjunto de test **neutral** (clase no vista en entrenamiento), mostrando que los modelos están más inseguros ahí (señal razonable de comprensión parcial de la ambigüedad).
* Usa **similitud coseno** para interpretar tweets mal clasificados (busca sus vecinos más parecidos en el train).
* Guarda el vectorizador y el mejor modelo entrenado (`tfidf\_vectorizer.pkl`, `lr\_model.pkl`) y la tabla comparativa (`resultados\_modelos.csv`).



**Resultado principal:** Regresión Logística (\~77.7% accuracy) > Naive Bayes (\~76.6%) > TextBlob pre-entrenado (\~74.4%) sobre el test externo.

### 

### &#x09;03\_topicos\_embeddings\_visualizaciones.ipynb



* Keywords por clase (TF-IDF diferencial).
* **BERTopic** para modelado de tópicos no supervisado, con **dendrograma jerárquico** (scipy, sobre la matriz c-TF-IDF). *Nota metodológica: se usa un Word2Vec propio como backend de embeddings en vez del sentence-transformer por defecto de BERTopic, porque este entorno no tiene acceso a Hugging Face Hub — documentado en detalle dentro del notebook.*
* **Juego de analogías** con embeddings: Word2Vec propio (corpus chico, resultados limitados) vs. GloVe pre-entrenado en 2B tweets (resultados coherentes). Incluye función de menú reutilizable para probar analogías propias.
* **PMI** (Pointwise Mutual Information) para encontrar bigramas característicos por clase.
* **Análisis de grafos**: red de co-ocurrencia de palabras (aristas ponderadas por PMI), con centralidad de grado y detección de comunidades, por separado para tweets positivos y negativos.
* **UMAP**: proyección 2D de los embeddings de tweets, coloreada por sentimiento real.

\---

## 

## 5\. Limitaciones documentadas



* El train es **binario** (negativo/positivo); el test externo tiene una tercera clase (neutral) que ningún modelo entrenado puede predecir por diseño. Se aborda explícitamente en el notebook 2 en vez de ignorarse.
* BERTopic no usa un transformer de Hugging Face (bloqueado por red en este entorno) sino embeddings promedio de un Word2Vec propio — esto reduce la granularidad de los tópicos encontrados respecto a lo que se obtendría con un modelo pre-entrenado.
* El análisis temporal está aclarado en el notebook 1.

\---



## 6\. Conclusión General





El análisis confirma que el sentimiento en tweets es aprendible a partir del contenido textual con relativamente poco dato (5000 tweets alcanzaron para superar a un modelo pre-entrenado genérico), pero también deja claro que hay un techo de complejidad (ironía, ambigüedad, jerga específica) que un enfoque bag-of-words simple no resuelve del todo.



###### Hallazgos principales por notebook



###### 1\. EDA y preprocesamiento



El dataset original de 1.6M está ordenado por clase.

Ni la longitud del tweet, ni la hora del día, ni el día de la semana correlacionan de forma relevante con el sentimiento (matriz de correlación con valores cercanos a 0) — el sentimiento no tiene atajos estructurales, depende del contenido.

Bug real encontrado y corregido: entidades HTML sin decodificar (\&lt;3) contaminaban el vocabulario con fragmentos basura (lt, gt).



###### 2\. Modelado de sentimiento



Regresión Logística (\~77.7% accuracy) > Naive Bayes (\~76.6%) > TextBlob pre-entrenado (\~74.4%) sobre el test externo — un modelo chico pero ajustado al dominio (tweets) generaliza mejor que un lexicón genérico.

Los modelos, entrenados en binario, quedan "más inseguros" (probabilidades más cercanas a 0.5) frente a los tweets que un humano etiquetó como neutrales — evidencia indirecta de que sí captan algo de la ambigüedad real, aunque no puedan nombrarla.

La similitud coseno mostró que buena parte de los errores se explican por vocabulario poco representado en el train (nombres propios, jerga 2009) o por sentimiento dependiente de contexto/ironía que TF-IDF no captura.



###### 3\. Tópicos, embeddings, grafos y visualizaciones



BERTopic encontró un tópico dominante genérico + 2-3 tópicos minoritarios interpretables (ceremonias/eventos, miedo/aburrimiento, quejas cotidianas) — la granularidad quedó limitada por usar embeddings propios (Word2Vec) en vez de un transformer, decisión forzada por falta de acceso a Hugging Face en el entorno.

El tamaño del corpus de embeddings importa mucho más que el algoritmo: el Word2Vec propio (5000 tweets) dio analogías poco coherentes, mientras que GloVe pre-entrenado en 2B tweets resolvió analogías razonablemente bien (good-bad+terrible→horrible, king-man+woman funcionando).

PMI y el grafo de co-ocurrencia (con detección de comunidades) llegaron, por caminos distintos, a patrones temáticos parecidos a los de BERTopic — dos técnicas independientes convergiendo es una buena señal de robustez.

UMAP mostró que positivos y negativos se mezclan bastante en el espacio de embeddings, consistente con el \~77% de accuracy: hay señal real, pero no separación perfecta — esperable en texto corto e informal.



###### 4\. Limitaciones encontradas



Clase neutral no vista, embeddings propios más débiles que un pretrained, ambigüedad/ironía no resuelta, no son fallas del análisis y quedaron documentadas explícitamente en cada notebook.



