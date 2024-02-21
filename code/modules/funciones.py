
import pandas as pd
import numpy as np
from glob import glob
import os
from importlib.machinery import SourceFileLoader
import xlsxwriter

# import variables_nombres as vn
# vn  = SourceFileLoader( 'variables_nombres', 'variables_nombres.py' ).load_module()

import variables_nombres as vn


from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score, matthews_corrcoef, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold

import pickle
import joblib
import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from collections import Counter


#import datawig
#from datawig.utils import random_split
#from datawig import SimpleImputer






####################################
# ÍNDICE
# 1. IMPORTACIÓN Y PREPROCESAMIENTO
# 2. FILTROS: SELECCIÓN DE VARIABLES
# 3. MÉTODOS DE IMPUTACIÓN
# 4. TRANSFORMACIONES
# 5. IMPLEMENTACION DE MODELOS
####################################
       

    
    
    
    
# 1. IMPORTACIÓN Y PREPROCESAMIENTO
###################################
    
def contar_variables( datasets ):
    '''
    Propósito:
        - Determinar cuántas variables pertenecen a SIAF y cuantas a RENAMU
    Inputs:
        - Lista de bases indexada
    Outputs:
        - Oraciones por cada una de las bases de la lista de bases,
          que especifican el número de variables de Renamu y SIAF
    Especificaciones:
        - Las bases deben estar nombradas previamente. Ejemplo:
          bases[0].name = 'base0'
    '''
    renamu_vars = []
    siaf_vars = []
    for dataset in datasets:
        renamu_vars = [ var for var in dataset.columns if var in vn.renamu_variables ]
        renamu_count = len( renamu_vars )
        siaf_vars = [ var for var in dataset.columns if var in vn.siaf_variables ]
        siaf_count = len( siaf_vars )
        politica_vars = [ var for var in dataset.columns if var in vn.politica_variables ]
        politica_count = len( politica_vars )
        
        print(f'{ dataset.name }: Variables de Renamu: { renamu_count }; Variables de SIAF: { siaf_count }; variables políticas { politica_count }')
        
        
def listar_variables( dataset, df = "siaf" ):
    '''
    Propósito:
        - Generar una lista de variables pertenecientes, o bien a SIAF o bien
          a Renamu, dada la base usada.
    Imputs:
        - dataset: dataset cuyas variables pertenecientes a SIAF o a Renamu quiere
          conocerse. A diferencia de otras funciones, se trata de una dataset, no
          de una lista de variables.
        - df: booleano. Si es siaf, se genera la lista de variables pertenecientes
          a SIAF. Si es renamu, se genera la lista de variables pertenecientes a
          Renamu. Si es politica, se genera la lista de variables pertenecientes a
          politica.
    Output:
        - Lista de variables de la dataset usada, que pertenecen a SIAF/Renamu, según
          lo espeficado en el segundo argumento de la función.
    '''    
    if df == "siaf": 
        variables = [ var for var in dataset.columns if var in vn.siaf_variables ]
    if df == "renamu":
        variables = [ var for var in dataset.columns if var in vn.renamu_variables ]
    if df == "politica":
        variables = [ var for var in dataset.columns if var in vn.politica_variables ]
        
    return variables    
    
    
def importar_bases( ruta ):
    '''
    Propósito:
        - Abre multiples datasets al mismo tiempo
    Inputs:
        - La ruta de acceso donde se encuentra las datasets
    Outputs:
        - La función crea una lista de bases indexada.
          Para acceder a acada una de las bases solo hay que 
          llamar a la lista con el índice correspondiente.
          Por ejemplo: lista_bases[0]
    Especificaciones:
        - Las datasets se encuentran en formato .dta
        - Si que quiere abrir todas las datasets de una determinada
          carpeta, se adiciona la siguiente expresión a la ruta de
          acceso de la carpeta "*.dta". 
    '''
    ruta_de_acceso = glob( ruta )
    lista_bases = []
    for i, base in enumerate( ruta_de_acceso ):
        dataframes = pd.read_stata( base,
                                    convert_categoricals = False,
                                    convert_dates = False,
                                    convert_missing = True )
        lista_bases.append( dataframes )
        
    return lista_bases



def determinar_dimensiones( dataframes ):
    '''
    Propósito:
        - Determinar las dimensiones de cada una de las bases
          de datos pertenecientes a una lista de bases
    '''
    for dataframe in dataframes:
        
        print( dataframe.name, dataframe.shape )
        
                 
def generar_tablas_descriptivas( bases, filename, ruta_output ):
    
    '''
    Propósito:
        - Generar tablas descriptivas para cada una de las bases de datos
          de un path brindado.
    '''
    
    export_path = os.path.join( ruta_output, f'{ filename }.xlsx' )
    
    writer      = pd.ExcelWriter( export_path, engine =  'xlsxwriter' )
    
    for i, base in enumerate( bases ):
        n_obs = bases[ i ].shape[ 0 ]
        desc = bases[ i ].describe( include = 'all' ).T
        desc[ 'missing_values_count' ] = n_obs - desc[ 'count' ]
        desc[ 'missing_values_percentage' ] = round( desc[ 'missing_values_count' ] / n_obs, 2 )
        desc = desc.sort_values( by = "missing_values_percentage",
                                 ascending = False )

        fuente = []
        
        for index, row in desc.iterrows():
            if index in vn.siaf_variables: fuente.append( "SIAF" )
            elif index in vn.renamu_variables: fuente.append( "Renamu" )
            elif index in vn.politica_variables: fuente.append( "Politica" )
            else: fuente.append( "Other" )
        desc[ 'fuente' ] = fuente
        
        desc = desc[ [ 'missing_values_percentage', 'missing_values_count', 'fuente', 
                       'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max' ] ]
        desc = desc.round( 2 )  
        
        desc.to_excel( writer, sheet_name = f'caso_{ i + 1 }' )  
        
    writer.close()  
        
        

# 2. FILTROS: SELECCIÓN DE VARIABLES
####################################
        
def filtro_missings( dataframe, umbral ):
    '''
    Propósito: 
        - Aplicar un filtro de variables basado en el porcentaje
          de valores perdidos de cada columna. Al aplicar el filtro,
          permanecen las columnas con un porcentaje de valores perdidos
          menor al umbral asignado.
    Inputs:
        - dataframe: bases de datos
        - umbral: porcentaje de valores perdidos
    Output:
        - Base de datos filtrada de variables con valores perdidos igual o mayor
          al umbral espeficiado.
    '''
    df_modificada = dataframe.copy()
    df_modificada = df_modificada.loc[ : , df_modificada.isnull().mean() < umbral ]
    
    return df_modificada
    
    
def filtro_variabilidad( dataset, umbral, deps ):
    '''
    Propósito:
        - Aplicar un filtro de columnas (variables) basado en la variabilidad
          de las columnas. Al aplicar el filtro, permanecen las columnas con
          un porcentaje de variabilidad mayor al umbral asignado.
    Inputs:
        - dataset: bases de datos
        - umbral: umbral de variabilidad
        - deps: variables que no serán filtradas
    Output:
        - Base de datos filtrada de columnas con una variabilidad mayor o igual al
          umbral asignado.
    Especificaciones:
        - Esta función utiliza el algoritmo: sklearn.feature_selection.VarianceThreshold
        - La implementación del algoritmo se basa en el siguiente texto: 
          https://medium.com/nerd-for-tech/removing-constant-variables-feature-selection-463e2d6a30d9
    '''
    df_copiada           = dataset.copy()
    columnas_a_modificar = [ col for col in df_copiada.columns if col not in deps ]
    df_modificada        = df_copiada[ columnas_a_modificar ]
    
    var_thr = VarianceThreshold( threshold = umbral )
    var_thr.fit( df_modificada )
    columnas_a_quedarse = df_modificada.columns[ var_thr.get_support() ]
    columnas_a_borrar   = [col for col in df_modificada.columns if col not in columnas_a_quedarse]
    df_copiada          = df_copiada.drop( columnas_a_borrar, axis = 1 )
    
    return df_copiada


def filtro_correlacion( dataset, dep, umbral ):
    '''
    Propósito:
        - Aplicar un filtro de columnas (variables) basado en la correlación
          de columnas. Si la correlación entre dos variables predictoras supera
          el umbral, se elimina la variable que tiene menor correlación con la 
          variable dependiente.
    Inputs:
        - dataset: base de datos
        - dep: variable dependiente. Usualmente se llama así: 
          dataset['dep'].reset_index()
          umbral: umbral de correlación
    Output:
        - Base de datos en la que se eliminaron las variables descritas en el
          Propósito.
    Especificaciones:
        - A diferencia de las anteriores funciones, en este caso la función
          trabaja sobre una base de datos. El filtro debe repetirse por cada
          base de datos y por cada variable dependiente.
        - La función "corr" (calcula correlaciones) ignora por default aquellas
          filas con valores perdidos. 
    Créditos:
        - Basado en la función "low_corr_vars" creada por el profesor A. Cozzubo
          en: https://github.com/acozzubo/ML_Public_Policy
    '''
    df_modificada = dataset.copy()
    
    candidatas = df_modificada.columns.values.tolist()
    for var1 in df_modificada.columns.values.tolist():
        for var2 in df_modificada.columns.values.tolist():
            if var1 != var2 and var1 in candidatas and var2 in candidatas:
                if abs( df_modificada[ var1 ].corr( df_modificada[ var2 ] ) ) > umbral:
                    if abs( df_modificada[ var1 ].corr( dep ) ) <= abs( df_modificada[ var2 ].corr( dep ) ):
                        candidatas.remove( var1 )
                    else:
                        candidatas.remove( var2 )
    df_modificada = df_modificada[ candidatas ]
    
    return df_modificada


def filtro_vars( dataframe, path, plus_vars ):
    
    '''
    Propósito:
        Aplicar un filtro de variables para quedarse solo con aquellas variables
        que están presentes en la base de datos del conjunto de entrenamiento
        con el que fueron entrenados los modelos  
    Inputs:
        - dataframe: dataframe
        - path: path en el que se encuentra el archivo excel con la lista
          de variables del conjunto de entrenamiento
        - plus_vars: variables adicionales que se desea preservar
    Output:
        - Base de datos en la que se eliminaron las variables descritas en el
          Propósito.    
    Especificaciones:
        Debe tenerse en cuenta que la variable 'variable_gasto' solo se encuentra
        disponible para los datos de entrenamiento y no la totalidad de los datos.
        Por tanto, en la próxima ejecución debe quitarse como variable predictora.
    '''
    
    df_modificada = dataframe.copy()
    
    vars_total    = pd.read_excel( path )
    vars_total    = vars_total[ 'colname' ].tolist()
    vars_total    = vars_total + plus_vars
    vars_quedarse = [ var for var in dataframe.columns if var in vars_total ]
    
    df_modificada = df_modificada[ vars_quedarse ]
    
    return df_modificada    
        

# 3. MÉTODOS DE IMPUTACIÓN
##########################
        
def imputar_i( dataframe, vars, val, dummy = False ):
    '''  
    Propósito:
       -  Imputar con un valor asignado por el usuario una lista determinada
          de variables, con la posibilidad de generar variables dummies de
          control por cada variable imputada.
    Inputs:
        - dataframe: dataframe
        - vars: lista de variables a imputar
        - val: valor que se usará para imputar
        - dummy: booleano. True en caso se desee generar una variable dummy 
          de control por cada variable imputada. 
    Output:
        - Dataframe imputada en las columnas seleccionadas, con el valor especificado, 
          y con/sin variables dummy de control.
    Créditos:
        - Basado en la función "mv_treat" creada por el profesor A. Cozzubo
          en: https://github.com/acozzubo/ML_Public_Policy
    '''
    
    df_modificada = dataframe.copy()
    
    for var in df_modificada.columns:
        if var in vars: 
            if df_modificada[ var ].isnull().sum() > 0:
                df_modificada.loc[ df_modificada[ var ].isnull(), var ] = val
                if dummy == True:
                    df_modificada[ 'im_' + var ] = 0
                    df_modificada.loc[ df_modificada[ var ].isnull(), 'im_' + var ] = 1
                
    return df_modificada


def imputar_ii( dataframe, variables, num = True, dummy = True ):
    '''
    Propósito:
        - Imputar con media o moda las bases de datos de la lista
          de bases brindada, con la posibilidad de generar variables
          dummies de control por cada variable imputada.
    Inputs:
        - dataframe: bases de datos
        - variables: lista de variables sobre las que se realizará la imputación
        - num: booleano. Indica si la lista de variables a ser imputada es 
          numérica (se imputa con media) o categórica (se imputa con moda)
        - dummy: booleano. Si es True, se generan variables de control 
          por cada una de las variables imputadas.
        
    Output:
        - Lista de bases de datos imputada en el conjunto de variables 
          seleccionadas, sean provenientes de Renamu o SIAF, con el valor 
          especificado, y con/sin variables dummy de control.
    '''
    df_modificada = dataframe.copy()
    
    for var in df_modificada.columns:
        if df_modificada[ var ].isna().sum() > 0:
            if num:
                if var in variables:
                    media = df_modificada[ var ].mean()
                    df_modificada.loc[ :, var ] = df_modificada[ var ].fillna( media )                    
                    if dummy:
                        df_modificada[ 'im_' + var ] = 0
                        df_modificada.loc[ df_modificada[ var ].isnull(), 'im_' + var ] = 1
            else:
                if var in variables:
                    moda = df_modificada[ var ].mode()[ 0 ]
                    df_modificada.loc[ :, var ] = df_modificada[ var ].fillna( moda )
                    if dummy:
                        df_modificada[ 'im_' + var ] = 0
                        df_modificada.loc[ df_modificada[ var ].isnull(), 'im_' + var ] = 1
                        
    return df_modificada


# def imputar_iii(dataframe):
    # '''
    # Propósito:
        # - Imputar valores perdidos usando la libreria datawig
    # Inputs:
        # - dataframe: base de datos que se utilizará
    # Output:
        # - Base de datos con todas las columnas que tienen valores perdidos
        #    imputadas. Las columnas imputadas "x" ahora se llaman "x_imputed".
    # Especificaciones:
        # - La libreria datawig solo puede ser ejecutada con la versión de
          # Python 3.7. Por ello, se sugiere crear un nuevoenvironment como 
          # se explica en: https://towardsdatascience.com/imputation-of-missing-data-in-tables-with-datawig-2d7ab327ece2
        # - El algoritmo datawig funciona del siguiente modo: se divide los datos
          # en training y test set. Se determina una columna a ser imputada con base
          # a otras columnas escogidas por el investigador. Se genera un modelo para
          # la columna específica, y se predice sobre el test set. Si la columna a ser
          # imputada es "x", se genera una nueva columna denominada "x_imputed".
        # - La presente función opera del siguiente modo: se divide la data en
          # training y test set. Se genera un modelo imputador por cada columna que presente un número
          # de missings mayor a 1. Posteriormente se predice cada columna a 
          # ser imputada en el test set. Si hay n columnas a ser imputadas, se generan
          # n bases de datos. Luego, se hace una concatenación horizontal a
          # todas las bases creadas en el test set. Se repite el mismo procedimiento
          # sobre el training set. Posteriormente se realiza una concatenación vertical
          # para unificar las observaciones pertenecientes al training set y el test set.
          # Finalmente, se eliminan las variables duplicadas "x" 
          # (dado que las columnas "x_imputed" las reemplazan").
          # '''
    # df_train, df_test = random_split(dataframe, split_ratios=[ 0.9, 0.1 ])
    # input_cols = df_train.columns.values.tolist()
    # append_test = []
    # append_train = []
    # vars = [ var for var in df_train.columns if df_train[var].isnull().sum() > 0 ]
    # for var in vars:
        # imputer = SimpleImputer( input_columns = input_cols,
                                       # output_column = var,
                                       # output_path = 'imputar_iii' )
        # imputer.fit( train_df = df_train )
        # predicciones_test = imputer.predict( df_test )
        # predicciones_train = imputer.predict( df_train )
        # append_test.append( predicciones_test )
        # append_train.append( predicciones_train )
    # pred_test = pd.concat( append_test, axis = 1 )
    # pred_train = pd.concat( append_train, axis = 1 )
    # pred = pd.concat( [ pred_test, pred_train ], axis = 0 )
    # pred = pred.loc[ : , ~pred.columns.duplicated() ].copy()
    # pred = pred.drop( pred[ vars ], axis = 1 )
    # return pred


# 4. TRANSFORMACIONES
#####################
    
    
def imputar_outliers( dataframe, vars, percentil_superior ):
    '''
    Propósito:
        - Imputar los valores del "percentil_superior" asignado
          con 1 - "percentil_superior". Ejemplo: imputar los 
          valores del percentil superior 1% con el percentil 99%. 
    Inputs:
        - dataframes bases de datos
        - vars: variables que se busca imputar
        - percentil_superior: percentil asignado. Los valores que 
          sobrepasen este percentil serán imputados con 1 - "percentil_superior".
    Output:
        - Base de datos con aquellos valores del "percentil_superior" 
          asignado imputados con 1 - "percentil_superior". 
    Créditos:
        - Basado en la función "outliers_imputation" creada por el 
          profesor A. Cozzubo en: https://github.com/acozzubo/ML_Public_Policy
    '''

    df_modificada = dataframe.copy()
    
    for var in vars:
        if var in df_modificada.columns:
            perc = df_modificada[ var ].quantile( 1 - percentil_superior )
            df_modificada.loc[ df_modificada[ var ] > perc, var ] = perc
            
    return df_modificada



def dividir_variables_negativas(dataset, val, deps):
    '''
    Propósito: 
        - Dividir todas aquellas variables negativas en la dataset
          por el valor asignado en la función
    Inputs:
        - dataset: base de datos
        - val: valor por el que las columnas negativas de la base de datos
          será reemplazado
        - deps: variables que no serán modificadas
    Output:
        - Base de datos con todas sus columnas negativas divididas entre
          el valor especificado
    Especificaciones:
        - Valor usado en la investigación: 1_000_000
    '''

    df_modificada = dataset.copy()
    
    for column in df_modificada.columns:
        if column not in deps and (df_modificada[column] < 0).any():
            df_modificada[column] = df_modificada[column] / val  
        
    return df_modificada



def transformacion_log( dataset, vars, deps ):
    '''
    Propósito:
        - Realizar una transformación logarítmica a las variables
          provenientes de SIAF y a las variables dependientes numéricas.
    Inputs:
        - dataset: base de datos
        - vars: variables a ser transformadas logarítmicamente
        - deps: variables que no serán modificadas
    Output:
        - Base de datos con valores pertenecientes a variables
          de SIAF transformadas logarítmicamente.
    Especificaciones:
        - Para evitar que los valores transformados logarítmicamente tomen
          valores negativos, se suma 1 a todos los valores de las variables SIAF.
    '''
    
    df_modificada = dataset.copy()
    
    log_vars = []
    
    for column in df_modificada.columns:
        if column not in deps:
            if (df_modificada[column] > 0).all():
                log_vars.append(column)
    
    for var in log_vars:
        if var in df_modificada.columns: 
            df_modificada[var] = np.log(df_modificada[var] + 1).astype(float)
        
    return df_modificada
            


def drop_missing_rows( dataset, missing_vars ):
    
    '''
    Propósito:
        - Eliminar las filas con valores perdidos de todas aquellas columnas con valores
          perdidos en una dataframe
    Input:
        - dataset: Base de datos
    Output:
        - Base de datos modificada con las filas con valores perdidos eliminadas
    '''
    
    df_modificada = dataset.copy()
    
    missing_cols = df_modificada[ missing_vars ].isnull().any( axis = 1 )
    data_final   = df_modificada[ ~missing_cols ]
    
    return data_final


# 5. IMPLEMENTACIÓN DE MODELOS
##############################

def resampling( x_train, y_train ):
    
    '''
    Objetivo:
    
        - Implementar una serie de métodos de muestreo para balancear bases de 
          datos para objetivos de clasificación. 
        - Los métodos de muestreo implementados son, en el siguiente orden: SMOTE,
          SMOTE Tomek-Links y Naive Random Oversampling.
          
    Input: 
    
        - x_train: conjunto de datos de entrenamiento con variables predictoras
        - y_train: conjunto de datos de entrenamiento con la variable predicha
        
    Output:
    
        - lista de dataframes que sigue el siguiente orden: x_train SMOTE, 
          x_train SMOTE Tomek-Links, x_train NRO, y_train SMOTE, y_train
          SMOTE Tomek-Links y y_train NRO.
    '''
    
    

    smote       = SMOTE( random_state = 2023, sampling_strategy = 'all' )    
    smote_tomek = SMOTETomek( random_state = 2023, sampling_strategy = 'all' )
    nro         = RandomOverSampler( random_state = 2023 )   

    x_train_s, y_train_s     = smote.fit_resample( x_train, y_train )    
    x_train_st, y_train_st   = smote_tomek.fit_resample( x_train, y_train )
    x_train_nro, y_train_nro = nro.fit_resample( x_train, y_train )
    
    return x_train_s, x_train_st, x_train_nro, y_train_s, y_train_st, y_train_nro
    


def extract_suffix(name):
    '''
    Objetivo: 
        - Extrae el sufijo del nombre del conjunto de entrenamiento.
          Ejemplo: 'x_train_st' -> 'st'
    '''
    parts = name.split('_')
    if len(parts) > 2:
        return '_'.join(parts[2:])
    return 'original'




def test_models_classification( models, x_train_list, y_train_list, x_test, y_test, path_list, sufix ):
    
    '''
    Objetivo:
    
        - Implementar una serie de modelos de ML de clasificación, y obtener métricas de rendimiento,
          importancia/coeficientes de variables, resultados de grid search y los modelos entrenados
          óptimos.
        - La presente función permite ejecutar modelos tanto si se espefican parámetros de grid search
          como si no.
        - La presente función asume que se utilizan métodos de muestreo, por los distintos modelos 
          especificados se entrenan con varios conjuntos de entrenamiento, pero se evalúan con el mismo
          conjunto de prueba. Se asume que los métodos de muestreo usados son (en el orden especificado):
          Original, SMOTE, SMOTE Tomek-Links y Naive Random Oversampling. 
          
    Input:
    
        - models      : Diccionario que especifica los modelos y los parámetros de grid search (en caso se
                        deseen implementar ). 
        - x_train_list: Lista de conjuntos de entrenamiento con las variables predictoras. La lista debe
                        seguir el siguiente orden: Original, SMOTE, SMOTE Tomek-Links y Naive Random 
                        Oversampling. Ejemplo: x_train_list = [ x_train, x_train_s, x_train_st, x_train_nro ]
        - y_train_list: Lista de conjuntos de entrenamiento con la variable predicha. La lista debe
                        seguir el siguiente orden: Original, SMOTE, SMOTE Tomek-Links y Naive Random 
                        Oversampling. Ejemplo: y_train_list = [ y_train, y_train_s, y_train_st, y_train_nro ]
        - x_test      : Conjunto de prueba con las variables predictoras
        - y_test      : Cnjunto de prueba con la variable predicha
        - path_list   : Lista de paths donde se guardarán los archivo output. Se asume que la lista de paths
                        tendrá cuatro elementos, los cuales son: path en el que se guardan los modelos preentrenados
                        (primero), path en el que se guardan las métricas de desempeño (segundo), path en el que se
                        guardan las  listas de variables con importancia/coeficientes (tercero) y path en el que se 
                        guardan los resultados de grid search (cuarto). Se asume que se sigue el orden mencionado en 
                        paréntesis.
                        
    Output:
    
        - resultados               : Pandas dataframe con las métricas de los distintos modelos implementados.
        - modelos entrenados       : todos los modelos entrenados se guardan en formato joblib en el path 
                                     especificado
        - lista de variables       : Listas de variables que muestra la importancia (en el caso de los métodos 
                                     basados en árboles) o coeficientes (en el caso de los métodos lineares) de 
                                     las variables predictoras. Se muestran en formato de tabla. 
        - resultados de grid search: Detalles sobre el ajuste del algoritmo de Grid Search. Se muestran en 
                                     formato de tabla.
                                     
    Especificaciones:
    
        - Los resultados se muestran con los siguientes sufijos: o, s, st y st. Si se sigue el orden indicado 
          para los input x_train_list y y_train_list, los números tienen las siguientes equivalencias:
          
             * o  : Modelo entrenado con el conjunto de entrenamiento Original
             * s  : Modelo entrenado con el conjunto de entrenamiento SMOTE
             * st : Modelo entrenado con el conjunto de entrenamiento SMOTE Tomek-Links
             * nro: Modelo entrenado con el conjunto de entrenamiento Naive Random Oversampling
             
        - Cuando se realiza Grid Search, se utiliza la estrategia de Cross Validation K Fold con 5 Splits. 
    '''
                                   
    results = {
        
        'Model'             : [],
        
        'accuracy_train'    : [],
        'log_loss_train'    : [],
        'roc_auc_train'     : [],
        'f1_train'          : [],
        'f1_train_si'       : [],
        'f1_train_no'       : [],        
        'MCC_train'         : [],               
        
        'accuracy_test'     : [],
        'log_loss_test'     : [],
        'roc_auc_test'      : [],
        'f1_test'           : [],
        'f1_test_si'        : [],
        'f1_test_no'        : [],        
        'MCC_test'          : [],
        
        'Grid_Search_Params': []
        
    }
    
    columns   = [ 'no', 'si' ]
    
    for path in path_list:
        if not os.path.exists( path ):
            os.makedirs( path )
    
    for model_name, model_params in models.items():
        
        if 'model' in model_params:
            model = model_params[ 'model' ]
        else:
            raise ValueError( f'Model is not defined for { model_name }' )
        
        if 'grid_params' in model_params:
            grid_params = model_params[ 'grid_params' ]
        else:
            grid_params = None
            
            
        for index, ( x_train, y_train ) in enumerate( zip( x_train_list, y_train_list ) ):

            train_suffix = extract_suffix( x_train.name ) 
            pred_vars      = x_train.columns.to_list()
            variables_dict = {}

            if grid_params is not None:

                cv          = KFold( n_splits = 5, shuffle = True, random_state = 2023 )
                scoring     = 'f1_macro'
                grid_search = GridSearchCV( model, grid_params, cv = cv, scoring = scoring )

                grid_search.fit( x_train, y_train )
                
                results_gs  = pd.DataFrame( grid_search.cv_results_ )
                results_gs.to_excel( f'{ path_list[ 3 ] }/gs_{ sufix }_{ model_name }_{ train_suffix }.xlsx' )

                best_model  = grid_search.best_estimator_
                best_params = grid_search.best_params_

                y_pred_train_class = best_model.predict( x_train )
                y_pred_train_proba = best_model.predict_proba( x_train )[ :, 1 ]            

                y_pred_test_class  = best_model.predict( x_test )
                y_pred_test_proba  = best_model.predict_proba( x_test )[ :, 1 ]
                
                joblib.dump( best_model, f'{ path_list[ 0 ] }/model_{ sufix }_{ model_name }_{ train_suffix }.joblib' )

                if hasattr( best_model, 'feature_importances_' ):
                    
                    feature_importances = best_model.feature_importances_
                    vars_df             = pd.DataFrame( {'Var': pred_vars, 'Importance Score': feature_importances } )
                    vars_df             = vars_df.reindex( vars_df[ 'Importance Score' ].abs().sort_values( ascending = False ).index )
                    vars_df.to_excel( f'{ path_list[ 2 ] }/varlist_{ sufix }_{ model_name }_{ train_suffix }.xlsx' )

                elif hasattr( best_model, 'coef_' ):
                    
                    coefficients = best_model.coef_[ 0 ]
                    vars_df      = pd.DataFrame( {'Var': best_model.feature_names_in_, 'Coefficient': coefficients } )
                    vars_df      = vars_df.reindex( vars_df[ 'Coefficient' ].abs().sort_values( ascending = False ).index )
                    vars_df.to_excel( f'{ path_list[ 2 ] }/varlist_{ sufix }_{ model_name }_{ train_suffix }.xlsx' )

            else:
                model.fit( x_train, y_train )
                
                if model_name == 'Logistic Regression':

                    best_params  = 'No grid search'
                    
                else:
                    
                    best_params  = model.C_[ 0 ]                   

                y_pred_train_class = model.predict( x_train )
                y_pred_train_proba = model.predict_proba( x_train )[ :, 1 ]            

                y_pred_test_class  = model.predict( x_test )
                y_pred_test_proba  = model.predict_proba( x_test )[ :, 1 ]
                
                joblib.dump( model, f'{ path_list[ 0 ] }/model_{ sufix }_{ model_name }_{ train_suffix }.joblib' )

                coefficients  = model.coef_[ 0 ]
                vars_df       = pd.DataFrame( {'Var': model.feature_names_in_, 'Coefficient': coefficients } )
                vars_df       = vars_df.reindex( vars_df[ 'Coefficient' ].abs().sort_values( ascending = False ).index )
                vars_df.to_excel( f'{ path_list[ 2 ] }/varlist_{ sufix }_{ model_name }_{ train_suffix }.xlsx' )

            report_train      = classification_report( y_train, y_pred_train_class, target_names = columns, output_dict = True )
            report_test       = classification_report( y_test, y_pred_test_class, target_names = columns, output_dict = True )            
            
            accuracy_train    = accuracy_score( y_train, y_pred_train_class )
            log_loss_train    = log_loss( y_train, y_pred_train_class )
            roc_auc_train     = roc_auc_score( y_train, y_pred_train_proba )
            f1_score_train    = f1_score( y_train, y_pred_train_class, average = 'macro' )
            f1_score_train_si = report_train[ 'si' ][ 'f1-score' ]
            f1_score_train_no = report_train[ 'no' ][ 'f1-score' ]
            mcc_score_train   = matthews_corrcoef( y_train, y_pred_train_class )

            accuracy_test    = accuracy_score( y_test, y_pred_test_class )
            log_loss_test    = log_loss( y_test, y_pred_test_class )
            roc_auc_test     = roc_auc_score( y_test, y_pred_test_proba )
            f1_score_test    = f1_score( y_test, y_pred_test_class, average = 'macro' )
            f1_score_test_si = report_test[ 'si' ][ 'f1-score' ]
            f1_score_test_no = report_test[ 'no' ][ 'f1-score' ]
            mcc_score_test   = matthews_corrcoef( y_test, y_pred_test_class )

            results[ 'Model' ].append( f'{ model_name }_{ train_suffix }' )
            
            results[ 'accuracy_train' ].append( round( accuracy_train, 3 ) )            
            results[ 'log_loss_train' ].append( round( log_loss_train, 3 ) )
            results[ 'roc_auc_train' ].append( round( roc_auc_train, 3 ) )
            results[ 'f1_train' ].append( round( f1_score_train, 3 ) )
            results[ 'f1_train_si' ].append( round( f1_score_train_si, 3 ) )
            results[ 'f1_train_no' ].append( round( f1_score_train_no, 3 ) )
            results[ 'MCC_train' ].append( round( mcc_score_train, 3 ) )               
            
            results[ 'accuracy_test' ].append( round( accuracy_test, 3 ) )
            results[ 'log_loss_test' ].append( round( log_loss_test, 3 ) )
            results[ 'roc_auc_test' ].append( round( roc_auc_test, 3 ) )
            results[ 'f1_test' ].append( round( f1_score_test, 3 ) )
            results[ 'f1_test_si' ].append( round( f1_score_test_si, 3 ) )
            results[ 'f1_test_no' ].append( round( f1_score_test_no, 3 ) )   
            results[ 'MCC_test' ].append( round( mcc_score_test, 3 ) )   
            
            results[ 'Grid_Search_Params' ].append( best_params )       

            results_df = pd.DataFrame( results )
            results_df.to_excel(( f'{ path_list[ 1 ] }/results_{ sufix }_{ model_name }_{ train_suffix }.xlsx' ))               
        
    results_df_general = pd.DataFrame( results )
    results_df_general = results_df_general.sort_values( by = 'f1_test', ascending = False )

    return results_df_general



def test_regression_forest(models, x_train_list, y_train_list, x_test, y_test, path_list, sufix):

    '''
    Objetivo:

        - Implementar el modelo Regression Forest adaptado para una clasificación binaria

    Input:

        - models      : Diccionario que especifica el modelo de Regressión Forest y los parámetros de grid search.
        - x_train_list: Lista de conjuntos de entrenamiento con las variables predictoras. La lista debe
                        seguir el siguiente orden: Original, SMOTE, SMOTE Tomek-Links y Naive Random 
                        Oversampling. Ejemplo: x_train_list = [ x_train, x_train_s, x_train_st, x_train_nro ]
        - y_train_list: Lista de conjuntos de entrenamiento con la variable predicha. La lista debe
                        seguir el siguiente orden: Original, SMOTE, SMOTE Tomek-Links y Naive Random 
                        Oversampling. Ejemplo: y_train_list = [ y_train, y_train_s, y_train_st, y_train_nro ]
        - x_test      : Conjunto de prueba con las variables predictoras
        - y_test      : Cnjunto de prueba con la variable predicha
        - path_list   : Lista de paths donde se guardarán los archivo output. Se asume que la lista de paths
                        tendrá cuatro elementos, los cuales son: path en el que se guardan los modelos preentrenados
                        (primero), path en el que se guardan las métricas de desempeño (segundo), path en el que se
                        guardan las  listas de variables con importancia/coeficientes (tercero) y path en el que se 
                        guardan los resultados de grid search (cuarto). Se asume que se sigue el orden mencionado en 
                        paréntesis.

    Output:

        - resultados               : Pandas dataframe con las métricas de los distintos modelos implementados.
        - modelos entrenados       : todos los modelos entrenados se guardan en formato joblib en el path 
                                     especificado
        - lista de variables       : Listas de variables que muestra la importancia (en el caso de los métodos 
                                     basados en árboles) o coeficientes (en el caso de los métodos lineares) de 
                                     las variables predictoras. Se muestran en formato de tabla. 
        - resultados de grid search: Detalles sobre el ajuste del algoritmo de Grid Search. Se muestran en 
                                     formato de tabla.
                                     
    '''

    columns   = [ 'no', 'si' ]
    threshold_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)
    
    results = {
        'Model'             : [],
        
        'accuracy_train'    : [],
        'log_loss_train'    : [],
        'roc_auc_train'     : [],
        'f1_train'          : [],
        'f1_train_si'       : [],
        'f1_train_no'       : [],        
        'MCC_train'         : [],               
        
        'accuracy_test'     : [],
        'log_loss_test'     : [],
        'roc_auc_test'      : [],
        'f1_test'           : [],
        'f1_test_si'        : [],
        'f1_test_no'        : [],        
        'MCC_test'          : [],
        
        'Grid_Search_Params': []
        
    }

    for model_name, model_params in models.items():
        if 'model' not in model_params:
            raise ValueError(f'Model is not defined for {model_name}')

        model = model_params['model']
        grid_params = model_params.get('grid_params', None)

        for index, (x_train, y_train) in enumerate(zip(x_train_list, y_train_list)):
            train_suffix = extract_suffix( x_train.name ) 
            pred_vars      = x_train.columns.to_list()

            cv = KFold(n_splits=5, shuffle=True, random_state=2023)
            grid_search = GridSearchCV(model, grid_params, cv=cv, scoring='r2')
            grid_search.fit(x_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Export models
            joblib.dump( best_model, f'{ path_list[ 0 ] }/model_{ sufix }_{ model_name }_{ train_suffix }.joblib' )

            # Export Grid Search Results
            results_gs  = pd.DataFrame( grid_search.cv_results_ )
            results_gs.to_excel( f'{ path_list[ 3 ] }/gs_{ sufix }_{ model_name }_{ train_suffix }.xlsx' )

            # Export features importance
            feature_importances = best_model.feature_importances_
            vars_df             = pd.DataFrame( {'Var': pred_vars, 'Importance Score': feature_importances } )
            vars_df             = vars_df.reindex( vars_df[ 'Importance Score' ].abs().sort_values( ascending = False ).index )
            vars_df.to_excel( f'{ path_list[ 2 ] }/varlist_{ sufix }_{ model_name }_{ train_suffix }.xlsx' )

            y_pred_train_proba = best_model.predict(x_train)
            y_pred_test_proba = best_model.predict(x_test)

            for threshold in threshold_range:

                # Calcular predicciones y métricas para el threshold actual
                y_pred_train_class = (best_model.predict(x_train) >= threshold).astype(int)
                y_pred_test_class = (best_model.predict(x_test) >= threshold).astype(int)

                # Agregar resultados al diccionario
                report_train = classification_report( y_train, y_pred_train_class, target_names = columns, output_dict = True )
                report_test = classification_report( y_test, y_pred_test_class, target_names = columns, output_dict = True )  
                
                # Calcular métricas para el conjunto de entrenamiento
                accuracy_train = accuracy_score(y_train, y_pred_train_class)
                log_loss_train = log_loss(y_train, y_pred_train_proba)
                roc_auc_train = roc_auc_score(y_train, y_pred_train_proba)
                f1_score_train = f1_score(y_train, y_pred_train_class, average='macro')
                f1_score_train_si = report_train['si']['f1-score']
                f1_score_train_no = report_train['no']['f1-score']
                mcc_score_train = matthews_corrcoef(y_train, y_pred_train_class)

                # Calcular métricas para el conjunto de pruebas
                accuracy_test = accuracy_score(y_test, y_pred_test_class)
                log_loss_test = log_loss(y_test, y_pred_test_proba) 
                roc_auc_test = roc_auc_score(y_test, y_pred_test_proba)
                f1_score_test = f1_score(y_test, y_pred_test_class, average='macro')
                f1_score_test_si = report_test['si']['f1-score']
                f1_score_test_no = report_test['no']['f1-score'] 
                mcc_score_test = matthews_corrcoef(y_test, y_pred_test_class)

                # Actualizar el diccionario de resultados
                results[ 'Model' ].append( f'{threshold}_{ model_name }_{ train_suffix }' )
                
                results[ 'accuracy_train' ].append( round( accuracy_train, 3 ) )            
                results[ 'log_loss_train' ].append( round( log_loss_train, 3 ) )
                results[ 'roc_auc_train' ].append( round( roc_auc_train, 3 ) )
                results[ 'f1_train' ].append( round( f1_score_train, 3 ) )
                results[ 'f1_train_si' ].append( round( f1_score_train_si, 3 ) )
                results[ 'f1_train_no' ].append( round( f1_score_train_no, 3 ) )
                results[ 'MCC_train' ].append( round( mcc_score_train, 3 ) )               
                
                results[ 'accuracy_test' ].append( round( accuracy_test, 3 ) )
                results[ 'log_loss_test' ].append( round( log_loss_test, 3 ) )
                results[ 'roc_auc_test' ].append( round( roc_auc_test, 3 ) )
                results[ 'f1_test' ].append( round( f1_score_test, 3 ) )
                results[ 'f1_test_si' ].append( round( f1_score_test_si, 3 ) )
                results[ 'f1_test_no' ].append( round( f1_score_test_no, 3 ) )   
                results[ 'MCC_test' ].append( round( mcc_score_test, 3 ) )   
                
                results[ 'Grid_Search_Params' ].append( best_params )  

        # Convertir el diccionario de resultados a DataFrame
        results_df_total = pd.DataFrame(results)
        results_df_total = results_df_total.sort_values( by = 'f1_test', ascending = False )

    return results_df_total