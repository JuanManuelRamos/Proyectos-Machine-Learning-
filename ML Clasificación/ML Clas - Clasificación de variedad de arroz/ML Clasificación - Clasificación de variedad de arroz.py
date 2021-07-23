#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning - Problema de Clasificación
# 
# En este ejercicio partiremos de un dataset de prueba para su análisis y la posterior creación de un modelo de clasificación que tratará de estimar la clase de una variable dependiente en casos futuros.
# Se seguirán los pasos recomendados de análisis de los datos, preprocesado, división de los datos, creación de modelos y evaluación de los mismos.
# 
# Se usará **Jupyter Notebook** con **Python** como lenguaje de programación.
# 
# <br><br>**Juan Manuel Ramos Pérez**

# ========================================================================================================================

# ## 1. Librerías
# 
# Comenzamos estableciendo todas las librerías de Python que nos harán falta para la realización del ejercicio, ordenadas según su función.

# In[1]:


# Tratamiento de datos
# ========================================
import numpy as np
import pandas as pd


# Gráficos
# ========================================
from matplotlib import pyplot as plt
from matplotlib import style
import matplotlib.ticker as ticker
import seaborn as sns


# Preprocesado
# ========================================
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_selector
from sklearn.model_selection import train_test_split



# Modelos
# ========================================
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from sklearn.model_selection import GridSearchCV


# Métricas
# ========================================
from sklearn.metrics import accuracy_score
from sklearn. metrics import classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer


# In[2]:


# Configuración
# ========================================
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams.update({'figure.max_open_warning': 0})
pd.set_option('display.max_rows', None)


# ## 2. Carga del dataset
# 
# Cargamos el dataset en el proyecto. Este se puede encontrar disponible en [UCI](https://archive.ics.uci.edu/ml/datasets/Rice+%28Cammeo+and+Osmancik%29).
# 
# Este dataset está compuesto por 3810 instancias, y recoge diferentes características morfológicas de granos de arroz cultivados en Turquía, la variedad Osmancik y la variedad Cammeo. La idea es crear un modelo que sea capaz de predecir de forma precisa la variedad a la que pertenece un grano de arroz en función de las características mencionados.
# 
# Las columnas de las que se compone el dataset son las siguientes:
# * Area (Area): numérica
# * Perímetro (Perimeter): numérica
# * Longitud del eje mayor (Major Axis Length): numérica
# * Longitud del eje menor (Minor Axis Length): numérica
# * Excentricidad (Eccentricity): numérica
# * Área convexa (Convex Area): numérica
# * Extensión (Extent): numérica
# * Clase (Class): categórica
# 
# 
# La variable "Class" será nuestra variable respuesta, o variable dependiente, aquella que queremos que nuestro modelo estime en datos futuros. Al tratarse de una variable categórica estamos ante un problema de clasificación, en el que tenemos como variables independientes o predictores numéricos.

# In[31]:


df = pd.read_excel('data/Rice_Osmancik_Cammeo_Dataset.xlsx')
df.head(10)


# ## 3. Análisis de los datos
# 
# En esta parte del ejercicio nos centraremos en analizar los datos de los que disponemos. Comprobar su distribución en el dataset, averiguar si se están reconociendo de forma adecuada, duplicidad, datos faltantes, así como decidir si debemos realizar algún o algunos cambios sobre ellos en el apartado de preprocesado de cara a construir el modelo de predicción.

# #### 3.1. TIPOS DE DATOS
# 
# En primer lugar comprobamos el tipo de dato interpretado por el dataframe de Pandas, y verificamos que todo está como se espera.

# In[4]:


df.dtypes


# Todas las columnas tienen el tipo de dato esperado.

# #### 3.2. VALORES FALTANTES
# 
# Comprobamos si hay celdas vacías en alguno de los registros. 
# 
# Esta comprobación es importante, ya que la existencia de valores nulos o vacíos pueden llegar a tener un impacto negativo considerable en el entrenamiento del modelo. En caso de que los hubiese tendríamos que decidir de qué forma lidiar con ellos, o bien eliminando el registro completo del dataset o bien sustituyendo el valor vacío por otro.

# In[5]:


df.isna().sum().sort_values()


# No faltan datos

# #### 3.3. FILAS DUPLICADAS

# In[6]:


df[df.duplicated()]


# #### 3.4. SEPARACIÓN DE VARIABLES
# 
# Creamos listas para separar las variables independientes en características, o features, y la variable dependiente o label.
# 
# Esto nos va a permitir hacer referencia a ellas más adelante de forma más cómoda.

# In[7]:


features = ['AREA','PERIMETER','MAJORAXIS','MINORAXIS','ECCENTRICITY','CONVEX_AREA','EXTENT']
label = 'CLASS'


# #### 3.5. VARIABLE RESPUESTA

# In[8]:


sns.countplot(x=df['CLASS'], label="Count", palette  = "pastel",)
plt.show()


# #### 3.6. VARIABLES PREDICTORAS

# En este caso coincide que todas las variables predictoras son numéricas y la variable respuesta es categórica.

# In[9]:


df[features].describe()


# #### 3.6.1. GRÁFICAS DE DISTRIBUCIÓN PARA LAS VARIABLES NUMÉRICAS
# 
# Crearemos gráficas para visualizar la distribución de cada una de las variables en el dataset. Mediante este análisis podremos comprobar los valores que toman las variables en general. Gracias a esta observación se pueden tomar decisiones como prescindir de alguna variable, transformarla a categórica, etc.

# In[10]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
axes = axes.flat

for i, colum in enumerate(features):
    sns.histplot(
        data    = df,
        x       = colum,
        stat    = "count",
        kde     = True,
        color   = (list(plt.rcParams['axes.prop_cycle'])*2)[i]["color"],
        line_kws= {'linewidth': 2},
        alpha   = 0.3,
        ax      = axes[i]
    )
    axes[i].set_title(colum, fontsize = 7, fontweight = "bold")
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")
    
# Se eliminan los axes vacíos
for i in [7, 8]:
    fig.delaxes(axes[i])
    
fig.tight_layout()
plt.subplots_adjust(top = 0.9)
fig.suptitle('Distribución variables numéricas', fontsize = 10, fontweight = "bold");


# #### 3.6.2. DISTRIBUCIÓN DE LAS VARIABLES PREDICTORAS RESPECTO A LA VARIABLE RESPUESTA 
# 
# En este apartado comprobaremos la relación existente entre las variables indepEndientes con respecto a la variable dependiente. 

# In[11]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 20))
axes = axes.flat

for i, column in enumerate(features):
    sns.boxplot(
        data    = df,
        y       = column,
        x       = df['CLASS'],
        color   = (list(plt.rcParams['axes.prop_cycle'])*2)[i]["color"],
        saturation   = 0.5,
        ax      = axes[i]
    )
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("Class")
    
# Se eliminan los axes vacíos
for i in [7, 8]:
    fig.delaxes(axes[i])
    
fig.tight_layout()
plt.subplots_adjust(top = 0.9)
fig.suptitle('Distribución variables predictoras', fontsize = 10, fontweight = "bold");


# Mediante estas gráficas de cajas podemos observar que en general hay una buena correlación entre las variables predictoras y la variable respuesta. Quizás las dos variables en las que se visualiza una menor correlación es en la variable **Extent** y **Minor Axis**.

# #### 3.6.3. TRANSFORMAR LA VARIABLE RESPUESTA A NUMÉRICA
# 
# Para poder hacer un cálculo de correlación entre las variables predictoras y la variable respuesta más exhaustivo, transformaremos la variable respuesta de categórica a numérica, asignando el valor 0 a la variedad **Cammeo** y 1 a la variedad **Osmancik**.

# In[12]:


for x in df.index:
    if df.loc[x, "CLASS"] == 'Cammeo':
        df.loc[x, "CLASS"] = 0
    elif df.loc[x, "CLASS"] == 'Osmancik':
        df.loc[x, "CLASS"] = 1
       
df['CLASS'] = pd.to_numeric(df['CLASS'])
df.info()


# #### 3.6.4. CORRELACIÓN RESPECTO A LA VARIABLE RESPUESTA

# In[13]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 5))
axes = axes.flat

for i, colum in enumerate(features):
    sns.regplot(
        x           = df[colum],
        y           = df['CLASS'],
        color       = "gray",
        marker      = '.',
        scatter_kws = {"alpha":0.4},
        line_kws    = {"color":"r","alpha":0.7},
        ax          = axes[i]
    )
    axes[i].set_title(f"CLASS vs {colum}", fontsize = 7, fontweight = "bold")
    axes[i].yaxis.set_major_formatter(ticker.EngFormatter())
    axes[i].xaxis.set_major_formatter(ticker.EngFormatter())
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

# Se eliminan los axes vacíos
for i in [7, 8]:
    fig.delaxes(axes[i])
    
fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Correlación con CLASS', fontsize = 10, fontweight = "bold");


# In[14]:


corr = round(df.corr(), 3)
corr.style.background_gradient()


# In[15]:


corr = df.corr()
print(corr["CLASS"].sort_values(ascending=False))


# Como se había interpretado en los gráficos de cajas, las variables **Major Axis**, **Perimete** y **Convex Area** tienen una fuerte correlación con la variable respuesta. La variable **Eccentricity** posee algo de correlación, y las variables **Minor Axis** y **Extent** una correlación muy baja.

# ## 4. Preprocesado

# Estandarizamos las variables numéricas para que se midan en la misma escala. Utilizaremos el escalador *MinMax*, el cual otorga valores entre 0 y 1, siendo estos el valor mínimo y máximo respectivamente.

# In[32]:


scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])
df.head(10)


# ## 5. División de datos de entrenamiento y test
# 
# 
# #### 5.1. SEPARACIÓN DE DATOS
# Dadas las circunstancias de correlación entre variables independientes y respuesta, se opta por prescindir de las variables **minoraxis** y **extent**.
# 
# Seguidamente se hacen las particiones correspondientes a los grupos de entrenamiento y test, siendo el tamaño de los datos de entrenamiento de un 80% con respecto al dataset completo.

# In[17]:


X_train, X_test, y_train, y_test = train_test_split(
                                         df.drop(columns = ['MINORAXIS', 'EXTENT', 'CLASS'], axis = 'columns'),
                                         df[label],
                                         train_size      = 0.8,
                                         random_state    = 0
                                     )


print ('Datos de entrenamiento: %d\nDatos de test: %d' % (X_train.shape[0], X_test.shape[0]))


# In[18]:


print("Partición datos de entrenamento")
print("-----------------------")
print(y_train.describe())


# In[19]:


print("Partición datos de test")
print("-----------------------")
print(y_test.describe())


# #### 5.2. DISTRIBUCIÓN DE LA VARIABLE RESPUESTA
# 
# Comprobamos que la distribución de la variable respuesta sea similar en los sets de entrenamiento y test. Una distribución muy dispar entre ambos sets podría dar problemas al modelo a la hora de enfrentarlo a los datos de test.

# In[20]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
axes = axes.flat
dt = [y_train, y_test]
title = ["Entrenamiento", "Test"]

for i, d in enumerate(dt):
    sns.countplot(
        x        = d,
        color    = (list(plt.rcParams['axes.prop_cycle'])*2)[i]["color"],
        palette  = "pastel",
        ax       = axes[i]
    )
    axes[i].set_title(title[i], fontsize = 7, fontweight = "bold")
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")


    
fig.tight_layout()
plt.subplots_adjust(top = 0.9)
fig.suptitle('Distribución de la variable respuesta en el set de entrenamiento y test', fontsize = 10, fontweight = "bold");


# ## 6. Creación de un modelo

# ### LOGISTIC REGRESSION

# In[21]:


# Create classifier
model = LogisticRegression(random_state=0)
model = model.fit(X_train, y_train)
pred = model.predict(X_test)


# ## 7. Evaluación del modelo

# En primer lugar creamos los métodos que nos van a servir para calcular y mostrar diferentes marcadores y métricas de la calidad de los modelos.

# In[22]:


# PRECISIÓN DEL CLASIFICADOR y MATRIZ DE CONFUSIÓN

def get_accuracy_scores(y_test, predictions, clf_name):  
    print('%s accuracy score: %.4f' % (clf_name, accuracy_score(y_test, predictions)))
    print('Overall Precision: ', precision_score(y_test, predictions))
    print('Overall Recall: ', recall_score(y_test, predictions))
    print('Classification report:\n', classification_report(y_test, predictions))
    
    cm = confusion_matrix(y_test, predictions)
    print ('\n\nConfusion Matrix:\n', cm)


# In[23]:


# MATRIZ DE CONFUSIÓN y MATRIZ DE PROBABILIDADES

def get_pred_proba(X_test, y_test, predictions, clf):  
    y_scores = pd.DataFrame(data=clf.predict_proba(X_test))
    print('\nPredict proba:\n', y_scores) 


# In[24]:


# CURVA ROC

def get_roc(X_test, y_test, clf):
    
    
    # calculate ROC curve
    y_scores = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

    # plot ROC curve
    print('\n\nROC Curve')
    fig = plt.figure(figsize=(6, 6))
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    
    

    auc = roc_auc_score(y_test, y_scores[:,1])
    print('\nROC AUC Score: ' + str(auc))
    return auc


# In[25]:


# LOGISTIC REGRESSION

# Accuracy scores
get_accuracy_scores(y_test, pred, 'Logistic Regression')

# Predict proba
#get_pred_proba(X_test, y_test, pred, clf)

# ROC curve
auc_lr = get_roc(X_test, y_test, model)


# ## 8. Utilización de otros algoritmos e hiperparámetros
# 
# En esta sección se utilizarán diversos algoritmos de clasificación para crear diferentes modelos a los que se le aplicará la técnica de validación cruzada, o cross validation, para tratar de encontrar los mejores hiperparámetros de cada uno de ellos.

# ### K NEIGHBORS con GridSearchCV

# In[26]:


# modelo
model = KNeighborsClassifier()

# parámetros
params = {
 'n_neighbors' : [1, 5, 10, 15, 20, 25, 30],
 'weights'     : ['uniform', 'distance'] ,
 'algorithm'   : ['ball_tree', 'kd_tree', 'brute'] 
}

# gridsearchCV
gridsearch = GridSearchCV(model, params, cv=5)
gridsearch.fit(X_train, y_train)

# Evaluación del mejor modelo
model = gridsearch.best_estimator_
pred = model.predict(X_test)

# Métricas
print("Mejor parámetro:", gridsearch.best_params_, "\n")
get_accuracy_scores(y_test, pred, 'K Neighbors')
auc_kn = get_roc(X_test, y_test, model)


# ### DECISION TREE con GridSearchCV

# In[27]:


# modelo
model = tree.DecisionTreeClassifier()

# parámetros
params = {
 'max_depth': [10, 50, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_split': [10, 20, 30, 40, 50]
}

# gridsearchCV
gridsearch = GridSearchCV(model, params, cv=5)
gridsearch.fit(X_train, y_train)

# Evaluación del mejor modelo
model = gridsearch.best_estimator_
pred = model.predict(X_test)

# Métricas
print("Mejor parámetro:", gridsearch.best_params_, "\n")
get_accuracy_scores(y_test, pred, 'Decision Tree')
auc_dt = get_roc(X_test, y_test, model)


# ### SUPPORT VECTOR MACHINE con GridSearchCV

# In[28]:


# modelo
model = svm.SVC()

# parámetros
params = {
 'C'          : [0.1, 1, 10, 100], 
 'gamma'      : [1, 0.1, 0.01, 0.001],
 'kernel'     : ['rbf', 'poly', 'sigmoid'],
 'probability': [True]
}

# gridsearchCV
gridsearch = GridSearchCV(model, params, cv=5)
gridsearch.fit(X_train, y_train)

# Evaluación del mejor modelo
model = gridsearch.best_estimator_
pred = model.predict(X_test)

# Métricas
print("Mejor parámetro:", gridsearch.best_params_, "\n")
get_accuracy_scores(y_test, pred, 'SVM')
auc_svm = get_roc(X_test, y_test, model)


# ### GAUSSIAN NAIVE BAYES con GridSearchCV

# In[29]:


# modelo
model = GaussianNB()

# parámetros
params = {
 'var_smoothing': np.logspace(0,-9, num=100)
}

# gridsearchCV
gridsearch = GridSearchCV(model, params, cv=5)
gridsearch.fit(X_train, y_train)

# Evaluación del mejor modelo
model = gridsearch.best_estimator_
pred = model.predict(X_test)

# Métricas
print("Mejor parámetro:", gridsearch.best_params_, "\n")
get_accuracy_scores(y_test, pred, 'SVM')
auc_gnb = get_roc(X_test, y_test, model)


# ## 9. Comparativa de resultados
# 
# En esta última parte se compararán los resultados de la evaluación de cada uno de los modelos creados anteriormente.

# In[30]:


auc = pd.DataFrame({
                        'modelo': ['Linear Regression', 'K Neighbors', 'Support Vector Machine', 'Decision Tree', 'Gaussian Naive Bayes'],
                        'auc': [auc_lr, auc_kn, auc_svm, auc_dt, auc_gnb]
                     })
auc = auc.sort_values('auc', ascending=True)


fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))


axe.hlines(auc.modelo, xmin=0, xmax=auc.auc)
axe.plot(auc.auc, auc.modelo, "o", color='black')
axe.tick_params(axis='y', which='major', labelsize=12)
axe.set_title('ROC AUC Score')

for i, val in enumerate(auc.auc):
    axe.text(0.91, i+0.05, "%.4f" % val)


fig.tight_layout()


# ### FUENTES
# 
# * Creación de modelos de Machine Learning: https://docs.microsoft.com/es-es/learn/paths/create-machine-learn-models/
