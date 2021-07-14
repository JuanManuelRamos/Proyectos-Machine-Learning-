#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning - Problema de Regresión
# 
# En este ejercicio partiremos de un dataset de prueba para su análisis y la posterior creación de un modelo de regresión que tratará de estimar el valor de una variable dependiente en casos futuros.
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
import matplotlib
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
from sklearn.metrics import accuracy_score
from sklearn. metrics import classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# Modelos
# ========================================
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV


# Métricas
# ========================================
from sklearn.metrics import mean_squared_error, make_scorer, r2_score


# In[2]:


# Configuración
# ========================================
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams.update({'figure.max_open_warning': 0})
pd.set_option('display.max_rows', None)


# ## 2. Carga del dataset
# 
# Cargamos el dataset en el proyecto. Este se puede encontrar disponible en [Kaggle](https://www.kaggle.com/mirichoi0218/insurance) y [Github](https://github.com/stedy/Machine-Learning-with-R-datasets).
# 
# Este dataset está compuesto por 1338 filas, y recoge diferentes datos personales y médicos de clientes con el objetivo de establecer la cantidad económica a percibir por parte de un seguro. La idea es crear un modelo que sea capaz de predecir de forma precisa los costes del seguro que se le aplicarán a un cliente en función de los datos mencionados.
# 
# Las columnas de las que se compone el dataset son las siguientes:
# * Edad (Age): numérica
# * Sexo (Sex): categórica
# * Índice de Masa Coporal (BMI): numérica
# * Número de hijos (Children): numérica
# * Fumador (Smoker): categórica
# * Región (Region): categórica
# * Cargos del seguro (Charges): numérica
# 
# 
# La variable "charges" será nuestra variable respuesta, o variable dependiente, aquella que queremos que nuestro modelo estime en datos futuros. Al tratarse de una variable numérica continua estamos ante un problema de regresión, en el que tenemos como variables independientes o predictores tanto variables numéricas como categóricas.

# In[3]:


df = pd.read_csv('data/insurance.csv')
df


# ## 3. Análisis de los datos
# 
# En esta parte del ejercicio nos centraremos en analizar los datos de los que disponemos. Comprobar su distribución en el dataset, averiguar si se están reconociendo de forma adecuada, duplicidad, datos faltantes, así como decidir si debemos realizar algún o algunos cambios sobre ellos en el apartado de preprocesado de cara a construir el modelo de predicción.

# En primer lugar comprobamos el tipo de dato interpretado por el dataframe de Pandas, y verificamos que todo está como se espera.

# In[4]:


df.dtypes


# Comprobamos si hay celdas vacías en alguno de los registros. 
# 
# Esta comprobación es importante, ya que la existencia de valores nulos o vacíos pueden llegar a tener un impacto negativo considerable en el entrenamiento del modelo. En caso de que los hubiese tendríamos que decidir de qué forma lidiar con ellos, o bien eliminando el registro completo del dataset o bien sustituyendo el valor vacío por otro.

# In[5]:


df.isna().sum().sort_values()


# Comprobamos si hay filas duplicadas.
# 
# En este caso identificamos una fila duplicada. Al tratarse solo de una observación procederemos a eliminarla del dataset.

# In[6]:


df[df.duplicated()]


# In[7]:


df = df.drop(df.index[581])


# Creamos listas para separar las variables independientes en características, o features, y éstas a su vez en numéricas y categóricas, y la variable dependiente o label.
# 
# Esto nos va a permitir hacer referencia a ellas más adelante de forma más cómoda.

# In[8]:


featuresNum = ['age', 'bmi', 'children']
featuresCat = ['sex', 'smoker', 'region']
label = 'charges'


# #### VARIABLE RESPUESTA
# 
# Ya que nuestra variable dependiente, o variable respuesta, se trata de una variable numérica continua, resulta interesante observar el tipo de distribución que tiene en el dataset.
# 
# En función su distribución, algunos modelos de machine learning podrán ajustarse mejor o no.

# In[9]:


fig, axes = plt.subplots()
sns.kdeplot(df.charges, 
            shade=True)


fig.tight_layout()


# En el siguiente fragmento se hará una comparación de esta distribución con varias de las distribuciones más típicas, con el objetivo de averiguar cuál de ellas se ajusta más, a través del **Sum of Square Error (SSE)**, a la de nuestra variable respuesta.
# 
# Fuente: [stackoverflow](https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python?lq=1)

# In[10]:


titulo1 = u'Charges.\n Todas las distribuciones ajustadas'
xlabel = u'Frecuencia'
ylabel = 'charges'

titulo2 = u'Charges con la mejor distribución.\n'

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings

import scipy.stats as st
import statsmodels.api as sm


matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check  
    DISTRIBUTIONS = [        
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm, st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

# Load data from statsmodels datasets
data = df.charges

# Plot for comparison
plt.figure(figsize=(12,8))
#ax = data.plot(kind='hist', bins=50, normed=True, alpha=0.5, color=plt.rcParams['axes.color_cycle'][1])
ax = data.plot(kind='hist', bins=50, density=True, alpha=0.5, color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'])
# Save plot limits
dataYLim = ax.get_ylim()

# Find best fit distribution
best_fit_name, best_fit_params = best_fit_distribution(data, 200, ax)
best_dist = getattr(st, best_fit_name)

# Update plots
ax.set_ylim(dataYLim)
ax.set_title(titulo1)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)

# Make PDF with best params 
pdf = make_pdf(best_dist, best_fit_params)

# Display
plt.figure(figsize=(12,8))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Datos', legend=True, ax=ax)

param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
dist_str = '{}({})'.format(best_fit_name, param_str)

ax.set_title(titulo2 + dist_str)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)


# Gracias a ello podemos comprobar que la distribución de nuestra variable respuesta se ajusta más a la de una **exGaussian distribution**, es decir, una distribución Gaussiana exponencialmente modificada.

# #### VARIABLES NUMÉRICAS
# 
# En este apartado haremos un análisis de las variables independientes numéricas. 

# En primer lugar echaremos un vistazo general.
# 
# Se puede comprobar que, a pesar de formar parte de variables completamente diferentes por definición, "age" y "bmi" comparten un rango de valores similar, no así la variable "children", la cual toma unos pocos valores.

# In[11]:


df[featuresNum].describe()


# Al igual que con la variable respuesta, hacemos una representación gráfica de la distribución de nuestras variables independientes numéricas.

# In[12]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
axes = axes.flat

for i, column in enumerate(featuresNum):
    sns.histplot(
        data    = df,
        x       = column,
        stat    = "count",
        kde     = True,
        color   = (list(plt.rcParams['axes.prop_cycle'])*2)[i]["color"],
        line_kws= {'linewidth': 2},
        alpha   = 0.3,
        ax      = axes[i]
    )
    axes[i].set_title(column, fontsize = 7, fontweight = "bold")
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")
    
fig.tight_layout()
plt.subplots_adjust(top = 0.9)
fig.suptitle('Distribución de variables predictoras numéricas', fontsize = 10, fontweight = "bold");


# In[13]:


df.children.value_counts().sort_index()


# Como ya se había comprobado desde la descripción de estas variables, y se corrobora en este último paso de forma más visual, la variable *children* toma unos pocos valores. 
# 
# No es recomendable tener una variable numérica que tome pocos valores, sobre todo si la mayoría de ellos se concentran en uno sólo de estos. Así que sería buena idea transformar esta variable a categórica. No sólo eso, sino además concentrar todas las observaciones que se hacen para número de hijos igual o mayor que 3 en una sóla categoría (*3_mas*), ya que las categorías 4 y 5 hijos apenas tienen representación.

# In[14]:


df.children = df.children.astype("str")


# In[15]:


df.dtypes


# In[16]:


children_replace = {'3': "3_mas", '4': "3_mas", '5': "3_mas"}

df['children'] = df['children'].map(children_replace).fillna(df['children'])


# In[17]:


df.children.value_counts().sort_index()


# De esta forma, dentro de las variables independientes, nos quedamos con dos variables numéricas y cuatro variables categóricas.
# 
# Actualizamos nuestras listas de variables.

# In[18]:


featuresNum = ['age', 'bmi']
featuresCat = ['sex', 'children', 'smoker', 'region']
label = 'charges'


# In[19]:


df[featuresNum].describe()


# In[20]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
axes = axes.flat

for i, column in enumerate(featuresNum):
    sns.histplot(
        data    = df,
        x       = column,
        stat    = "count",
        kde     = True,
        color   = (list(plt.rcParams['axes.prop_cycle'])*2)[i]["color"],
        line_kws= {'linewidth': 2},
        alpha   = 0.3,
        ax      = axes[i]
    )
    axes[i].set_title(column, fontsize = 7, fontweight = "bold")
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")
    
fig.tight_layout()
plt.subplots_adjust(top = 0.9)
fig.suptitle('Distribución de variables predictoras numéricas', fontsize = 10, fontweight = "bold");


# Calculamos la correlación entre las variables predictoras y la variable respuesta.

# In[21]:


corr = round(df.corr(), 3)
corr.style.background_gradient()


# In[22]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
axes = axes.flat

for i, column in enumerate(featuresNum):
    sns.regplot(
        x           = df[column],
        y           = df[label],
        color       = "gray",
        marker      = '.',
        scatter_kws = {"alpha":0.4},
        line_kws    = {"color":"r","alpha":0.7},
        ax          = axes[i]
    )
    axes[i].set_title(f"charges vs {column}", fontsize = 7, fontweight = "bold")
    #axes[i].ticklabel_format(style='sci', scilimits=(-4,4), axis='both')
    axes[i].yaxis.set_major_formatter(ticker.EngFormatter())
    axes[i].xaxis.set_major_formatter(ticker.EngFormatter())
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel(column)
    axes[i].set_ylabel(label)
    
fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Correlación con charges', fontsize = 10, fontweight = "bold");


# De estas operaciones sacamos algunas conclusiones:
# 
# De forma general, se considera que una correlación es fuerte a partir de un valor de $\pm 0.6 \sim \pm0.7$, y débil cuando está entre $0$ y $\pm3$. Por ello, la correlación entre las variables *age* y *bmi* respecto a *charges* es baja.
# 
# El punto negativo es que sería deseable que existiese una correlación alta entre las variables predictoras y la variable que queremos predecir. Si así fuera, nuestro modelo tendría más posibilidades de predecir dicha variable correctamente.
# 
# El punto positivo es que no es deseable, de cara a la fiabilidad del modelo, que exista una alta correlación entre las variables independientes. Y en este caso tampoco la hay entre las variables *age* y *bmi*.

# #### VARIABLES CATEGÓRICAS
# 
# Hacemos ahora una representación mediante gráficos de barras de la distribución de las variables categóricas.

# In[23]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
axes = axes.flat

for i, column in enumerate(featuresCat):
    df[column].value_counts().plot.barh(ax = axes[i], color = 'cadetblue')
    axes[i].set_title(column, fontsize = 7, fontweight = "bold")
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")
    
fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Distribución de variables categóricas',
             fontsize = 10, fontweight = "bold");


# Comprobamos que la proproción de valores en *sex* y *region* está muy igualada. Algo menos en la variable *children*, en la que la mayoría de los casos se los lleva un sólo valor. Y bastante más descompensada en *smoker*.

# Para hacernos una idea de la correlación entre las variables categóricas y la variable respuesta, mostraremos su distribución, en este caso mediante un gráfico de cajas y otro de violín.

# In[24]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
axes = axes.flat

for i, column in enumerate(featuresCat):
    sns.boxplot(
        data    = df,
        x       = column,
        y       = df['charges'],
        color   = (list(plt.rcParams['axes.prop_cycle'])*2)[i]["color"],
        ax      = axes[i]
    )
    #axes[i].set_title(column, fontsize = 7, fontweight = "bold")
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel(column)
    axes[i].set_ylabel("charges")
    
    
fig.tight_layout()
plt.subplots_adjust(top = 0.9)
fig.suptitle('Distribución de variables categóricas', fontsize = 10, fontweight = "bold");


# In[25]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
axes = axes.flat

for i, column in enumerate(featuresCat):
    sns.violinplot(
        data  = df,
        x     = column,
        y     = label,
        color = "white",
        ax    = axes[i]
    )
    #axes[i].set_title(f"precio vs {column}", fontsize = 7, fontweight = "bold")
    axes[i].yaxis.set_major_formatter(ticker.EngFormatter())
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel(column)
    axes[i].set_ylabel("charges")
    
fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Distribución del precio por variables categóricas', fontsize = 10, fontweight = "bold");


# Después de visualizar de forma gráfica la distribución de las variables categóricas se puede deducir que la correlación de *charges* respecto a las variables *sex*, *children* y *region* es prácticamente inexistente, ya que para todos los valores posibles de estas variables se toman prácticamente los mismos valores de *charges*. No así en cuanto a la variable *smoker*, en la que sí que se observa una clara diferencia del valor de *charges* cuando se trata de un fumador o un no fumador.

# Para tener un valor exacto de correlación, y con la idea de crear posteriormente el modelo de predicción, convertiremos todas las variables categóricas en numéricas binarizándolas mediante el método *get_dummies*.

# In[26]:


df2 = pd.get_dummies(df, columns= featuresCat)
df2.info()


# In[27]:


corr = df2.corr()
print(corr[label].sort_values(ascending=False))


# Ahora sí, tenemos unos resultados completos de correlación de todas las variables independientes con respecto a la variable dependiente.
# 
# De esta lista podemos observar que la variable que más nos va a ayudar a la hora de predecir datos futuros es *smoker*. Sería deseable contar con un mayor número de variables independientes con correlaciones fuertes para tener un modelo de preodicción más sólido.
# 
# El análsis de los datos es importante para observar este tipo de sucesos y valorar la posibilidad de hacer modificaciones en los mismos, o descartar directamente las variables que no nos vayan a aportar información útil.

# En las líneas posteriores se ha optado por unificar los valores de la variable *children*, pasando a tener únicamente los valores *0* hijos y *1_mas*. No obsante no se ha conseguido ni aún así obtener una correlación aceptable.

# In[28]:


children_replace = {'1': "1_mas", '2': "1_mas", '3_mas': "1_mas"}

df['children'] = df['children'].map(children_replace).fillna(df['children'])

df.children.value_counts().sort_index()


# In[29]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
axes = axes.flat

for i, column in enumerate(featuresCat):
    df[column].value_counts().plot.barh(ax = axes[i], color = 'cadetblue')
    axes[i].set_title(column, fontsize = 7, fontweight = "bold")
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")
    
fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Distribución de variables categóricas',
             fontsize = 10, fontweight = "bold");


# In[30]:


df3 = pd.get_dummies(df, columns= featuresCat)
df3.info()


# In[31]:


corr = df3.corr()
print(corr[label].sort_values(ascending=False))


# ## 4. Preprocesado

# Estandarizamos las variables numéricas para que se midan en la misma escala. Utilizaremos el escalador MinMax, el cual otorga valores entre 0 y 1, siendo estos el valor mínimo y máximo respectivamente.

# In[32]:


scaler = MinMaxScaler()
df[featuresNum] = scaler.fit_transform(df[featuresNum])
df


# In[33]:


df = pd.get_dummies(df, columns= featuresCat)
df.info()


# ## 5. División de datos de entrenamiento y test
# 
# Dadas las circunstancias de correlación entre variables independientes y respuesta, se opta por prescindir de las variables *sex*, *children* y *region*.
# 
# Seguidamente se hacen las particiones correspondientes a los grupos de entrenamiento y test, siendo el tamaño de los datos de entrenamiento de un 80% con respecto al dataset completo.

# In[34]:


X_train, X_test, y_train, y_test = train_test_split(
                                        df.drop(columns = ['charges', 'sex_female', 'sex_male', 
                                                           'children_0', 'children_1_mas', 'region_northeast',
                                                           'region_northwest', 'region_southeast', 'region_southwest'], axis = 'columns'),
                                        df[label],
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )


# In[35]:


print("Partición datos de entrenamento")
print("-----------------------")
print(y_train.describe())


# In[36]:


print("Partición datos de test")
print("-----------------------")
print(y_test.describe())


# Comprobamos que la distribución de la variable respuesta sea similar en los sets de entrenamiento y test. Una distribución muy dispar entre ambos sets podría dar problemas al modelo a la hora de enfrentarlo a los datos de test.

# In[37]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
axes = axes.flat
dt = [y_train, y_test]
title = ["Entrenamiento", "Test"]

for i, d in enumerate(dt):
    sns.histplot(
        data    = d,
        #x       = lb,
        stat    = "count",
        kde     = True,
        color   = (list(plt.rcParams['axes.prop_cycle'])*2)[i]["color"],
        line_kws= {'linewidth': 2},
        alpha   = 0.3,
        ax      = axes[i]
    )
    axes[i].set_title(title[i], fontsize = 7, fontweight = "bold")
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")


    
fig.tight_layout()
plt.subplots_adjust(top = 0.9)
fig.suptitle('Distribución de la variable respuesta en el set de entrenamiento y test', fontsize = 10, fontweight = "bold");


# ## 6. Creación de un modelo

# Comenzamos en primer lugar entrenando un modelo de regresión lineal. Posteriormente usaremos el modelo para predecir los cargos al seguro de nuestro conjunto de datos de test.

# #### REGRESIÓN LINEAL

# In[38]:


model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)


# ## 7. Evaluar el modelo

# Hacemos una comparativa gráfica entre los valores de cargo actuales y los valores predichos.

# In[39]:


plt.figure(figsize=(8,8))
plt.scatter(y_test, pred)
plt.xlabel('Valor actual')
plt.ylabel('Valor predicho')
plt.title('RL - Cargos del seguro (Charges)')

z = np.polyfit(y_test, pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), color='black')

plt.show()


# A pesar de que los resultados muestran una clara tendencia diagonal en los datos, hay un grupo de ellos en la zona central que no han sido predichos correctamente.
# 
# Para cuantificar el nivel de residuo en el modelo, calculamos la raíz del error cuadrático medio ($RMSE$) y el indicador $R^2$.

# In[40]:


mse = mean_squared_error(y_test, pred)
rmse_RL = np.sqrt(mse)
print("RMSE:", rmse_RL)

r2_RL = r2_score(y_test, pred)
print("R2:", r2_RL)


# ## 8. Utilización de otros algoritmos e hiperparámetros
# 
# En esta sección se utilizarán diversos algoritmos de regresión para crear diferentes modelos a los que se le aplicará la técnica de validación cruzada, o cross validation, para tratar de encontrar los mejores hiperparámetros de cada uno de ellos.

# En primer lugar, y ya que hemos comenzado con un modelo de regresión lineal, aplicaremos las correspondientes regularizaciones **Ridge** y **Lasso**.

# #### RIDGE con CV

# In[41]:


model = RidgeCV(alphas=[1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 20], store_cv_values=True).fit(X_train, y_train)
pred = model.predict(X_test)


# In[42]:


plt.clf()
plt.figure(figsize=(8,8))
plt.scatter(y_test, pred)
plt.xlabel('Valor actual')
plt.ylabel('Valor predicho')
plt.title('R - Cargos del seguro (Charges)')

z = np.polyfit(y_test, pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), color='black')

plt.show()


print("Mejor alpha:", model.alpha_)
print("CV scores: ", np.mean(model.cv_values_, axis=0))

mse = mean_squared_error(y_test, pred)
rmse_R = np.sqrt(mse)
print("RMSE:", rmse_R)

r2_R = r2_score(y_test, pred)
print("R2:", r2_R)


# #### LASSO con CV

# In[43]:


model = LassoCV(alphas=[1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3, 1e-2, 0.1, 1, 5, 10]).fit(X_train, y_train)
pred = model.predict(X_test)


# In[44]:


plt.clf()
plt.figure(figsize=(8,8))
plt.scatter(y_test, pred)
plt.xlabel('Valor actual')
plt.ylabel('Valor predicho')
plt.title('L - Cargos del seguro (Charges)')

z = np.polyfit(y_test, pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), color='black')

plt.show()


print("Mejor alpha:", model.alpha_)

mse = mean_squared_error(y_test, pred)
rmse_L = np.sqrt(mse)
print("RMSE:", rmse_L)

r2_L = r2_score(y_test, pred)
print("R2:", r2_L)


# Seguidamente se probarán con los algoritmos de regresión **K-Nearest Neighbors**, **Support Vector Machine**, **Decision Tree**, **Random Search** y **Gradient Boosting**. En todos ellos se utilizará **GridSearchCV**, una herramienta que implementa métodos de entrenamiento y evaluación dado un algoritmo y un conjunto de hiperparámetros que serán optimizados mediante validación cruzada.

# #### K-NEAREST NEIGHBORS REGRESSOR con GridSearchCV

# In[45]:


# K Neighbors Regressor con Grid Search CV

# modelo
model = KNeighborsRegressor()

# parámetros
params = {
 'n_neighbors': [1, 5, 10, 15, 20, 25, 30],
 'weights': ['uniform', 'distance']  
}

# métrica
score = make_scorer(r2_score)

# gridsearchCV
gridsearch = GridSearchCV(model, params, scoring=score, cv=5, return_train_score=True)
gridsearch.fit(X_train, y_train)

# Evaluación del mejor modelo
model = gridsearch.best_estimator_
pred = model.predict(X_test)


# In[46]:


plt.clf()
plt.figure(figsize=(8,8))
plt.scatter(y_test, pred)
plt.xlabel('Valor actual')
plt.ylabel('Valor predicho')
plt.title('KNN - Cargos del seguro (Charges)')

z = np.polyfit(y_test, pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), color='black')

plt.show()


print("Mejor parámetro:", gridsearch.best_params_, "\n")

mse = mean_squared_error(y_test, pred)
rmse_KN = np.sqrt(mse)
print("RMSE:", rmse_KN)

r2_KN = r2_score(y_test, pred)
print("R2:", r2_KN)


# #### SUPPORT VECTOR REGRESSION con GridSearch CV

# In[47]:


# Support Vector Regression con Grid Search CV

# modelo
model = SVR()

# parámetros
params = {
 'C': [0.1, 1, 10, 100], 
 'gamma': [1, 0.1, 0.01, 0.001],
 'epsilon': [0.1, 0.2],
 'kernel': ['rbf', 'poly', 'sigmoid']
}

# métrica
score = make_scorer(r2_score)

# gridsearchCV
gridsearch = GridSearchCV(model, params, scoring=score, cv=5, return_train_score=True)
gridsearch.fit(X_train, y_train)

# Evaluación del mejor modelo
model = gridsearch.best_estimator_
pred = model.predict(X_test)


# In[48]:


plt.clf()
plt.figure(figsize=(8,8))
plt.scatter(y_test, pred)
plt.xlabel('Valor actual')
plt.ylabel('Valor predicho')
plt.title('SV - Cargos del seguro (Charges)')

z = np.polyfit(y_test, pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), color='black')

plt.show()


print("Mejor parámetro:", gridsearch.best_params_, "\n")

mse = mean_squared_error(y_test, pred)
rmse_SV = np.sqrt(mse)
print("RMSE:", rmse_SV)

r2_SV = r2_score(y_test, pred)
print("R2:", r2_SV)


# #### DECISION TREE REGRESSOR con GridSearchCV

# In[49]:


# Decision Tree Regressor con Grid Search CV

# modelo
model = DecisionTreeRegressor()

# parámetros
params = {
 'max_depth': [10, 50, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_split': [10, 20, 30, 40, 50]
}

# métrica
score = make_scorer(r2_score)

# gridsearchCV
gridsearch = GridSearchCV(model, params, scoring=score, cv=5, return_train_score=True)
gridsearch.fit(X_train, y_train)

# Evaluación del mejor modelo
model = gridsearch.best_estimator_
pred = model.predict(X_test)


# In[50]:


plt.clf()
plt.figure(figsize=(8,8))
plt.scatter(y_test, pred)
plt.xlabel('Valor actual')
plt.ylabel('Valor predicho')
plt.title('DT - Cargos del seguro (Charges)')

z = np.polyfit(y_test, pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), color='black')

plt.show()


print("Mejor parámetro:", gridsearch.best_params_, "\n")

mse = mean_squared_error(y_test, pred)
rmse_DT = np.sqrt(mse)
print("RMSE:", rmse_DT)

r2_DT = r2_score(y_test, pred)
print("R2:", r2_DT)


# #### RANDOM FOREST REGRESSOR con GridSearchCV

# In[51]:


# Random Forest Regressor con Grid Search CV

# modelo
model = RandomForestRegressor()

# parámetros
params = {
 'max_depth': [10, 50, 100, None],
 'max_features': ['auto', 'sqrt'],
 'n_estimators': [100, 1000, 2000]
}

# métrica
score = make_scorer(r2_score)

# gridsearchCV
gridsearch = GridSearchCV(model, params, scoring=score, cv=5, return_train_score=True)
gridsearch.fit(X_train, y_train)

# Evaluación del mejor modelo
model = gridsearch.best_estimator_
pred = model.predict(X_test)


# In[52]:


plt.clf()
plt.figure(figsize=(8,8))
plt.scatter(y_test, pred)
plt.xlabel('Valor actual')
plt.ylabel('Valor predicho')
plt.title('RF - Cargos del seguro (Charges)')

z = np.polyfit(y_test, pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), color='black')

plt.show()


print("Mejor parámetro:", gridsearch.best_params_, "\n")

mse = mean_squared_error(y_test, pred)
rmse_RF = np.sqrt(mse)
print("RMSE:", rmse_RF)

r2_RF = r2_score(y_test, pred)
print("R2:", r2_RF)


# #### GRADIENT BOOSTING REGRESSOR con GridSearchCV

# In[53]:


# Gradient Boost Regressor con Grid Search CV

# modelo
model = GradientBoostingRegressor()

# parámetros
params = {
 'learning_rate': [0.1, 0.5, 1.0],
 'n_estimators' : [50, 100, 150]
 }

# métrica
score = make_scorer(r2_score)

# gridsearchCV
gridsearch = GridSearchCV(model, params, scoring=score, cv=5, return_train_score=True)
gridsearch.fit(X_train, y_train)

# Evaluación del mejor modelo
model = gridsearch.best_estimator_
pred = model.predict(X_test)


# In[54]:


plt.clf()
plt.figure(figsize=(8,8))
plt.scatter(y_test, pred)
plt.xlabel('Valor actual')
plt.ylabel('Valor predicho')
plt.title('GB - Cargos del seguro (Charges)')

z = np.polyfit(y_test, pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), color='black')

plt.show()


print("Mejor parámetro:", gridsearch.best_params_, "\n")

mse = mean_squared_error(y_test, pred)
rmse_GB = np.sqrt(mse)
print("RMSE:", rmse_GB)

r2_GB = r2_score(y_test, pred)
print("R2:", r2_GB)


# ## 9. Comparativa de resultados
# 
# En esta última parte se compararán los resultados de la evaluación de cada uno de los modelos creados anteriormente.

# In[55]:


error = pd.DataFrame({
                        'modelo': ['Linear Regression', 'Ridge', 'Lasso', 'K-NNeighbors', 'Support Vector', 'Decision Tree',
                                   'Random Forest', 'Gradient Boosting'],
                        'rmse': [rmse_RL, rmse_L, rmse_R, rmse_KN, rmse_SV, rmse_DT, rmse_RF, rmse_GB],
                        'r2': [r2_RL, r2_L, r2_R, r2_KN, r2_SV, r2_DT, r2_RF, r2_GB]
                     })
error = error.sort_values('rmse', ascending=False)


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

axes[0].hlines(error.modelo, xmin=0, xmax=error.rmse)
axes[0].plot(error.rmse, error.modelo, "o", color='black')
axes[0].tick_params(axis='y', which='major', labelsize=12)
axes[0].set_title('RMSE')
    
axes[1].hlines(error.modelo, xmin=0, xmax=error.r2)
axes[1].plot(error.r2, error.modelo, "o", color='black')
axes[1].tick_params(axis='y', which='major', labelsize=12)
axes[1].set_title('R2')

fig.tight_layout()


# ### FUENTES
# 
# * Machine learning con Python y Scikit-learn by Joaquín Amat Rodrigo, available under a Attribution 4.0 International (CC BY 4.0) at https://www.cienciadedatos.net/documentos/py06_machine_learning_python_scikitlearn.html
# 
# * Fitting empirical distribution to theoretical ones with Scipy (Python)?: https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python?lq=1
# 
# * Creación de modelos de Machine Learning: https://docs.microsoft.com/es-es/learn/paths/create-machine-learn-models/
