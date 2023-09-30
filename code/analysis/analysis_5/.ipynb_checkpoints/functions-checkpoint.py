import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold

import pandas as  pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from collections import Counter



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
    




def test_models_classification( models, x_train_list, y_train_list, x_test, y_test, path_list ):
    
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
                        tendrá tres elementos, los cuales son: path en el que se guardan los modelos preentrenados
                        (primero), path en el que se guardan las listas de variables con importancia/coeficientes 
                        (segundo) y path en el que se guardan los resultados de grid search (tercero). Se asume
                        que se sigue el orden mencionado en paréntesis.
                        
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
    
        - Los resultados se muestran con los siguientes números: 0, 1, 2 y 3. Si se sigue el orden indicado 
          para los input x_train_list y y_train_list, los números tienen las siguientes equivalencias:
          
             * 0: Modelo entrenado con el conjunto de entrenamiento Original
             * 1: Modelo entrenado con el conjunto de entrenamiento SMOTE
             * 2: Modelo entrenado con el conjunto de entrenamiento SMOTE Tomek-Links
             * 3: Modelo entrenado con el conjunto de entrenamiento Naive Random Oversampling
             
        - Cuando se realiza Grid Search, se utiliza la estrategia de Cross Validation K Fold con 5 Splits. 
    '''
                                   
    results = {
        
        'Model'             : [],
        'accuracy_train'    : [],
        'accuracy_test'     : [],
        'log_loss_train'    : [],
        'log_loss_test'     : [],
        'roc_auc_train'     : [],
        'roc_auc_test'      : [],
        'f1_train'          : [],
        'f1_test'           : [],
        'MCC_train'         : [],
        'MCC_test'          : [],
        'Grid_Search_Params': []
        
    }
    
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
            
            pred_vars      = x_train.columns.to_list()
            variables_dict = {}

            if grid_params is not None:

                cv          = KFold( n_splits = 5, shuffle = True, random_state = 2023 )
                grid_search = GridSearchCV( model, grid_params, cv = cv )

                grid_search.fit( x_train, y_train )
                
                results_gs  = pd.DataFrame( grid_search.cv_results_ )
                results_gs.to_excel( f'{ path_list[ 2 ] }/gs_{ model_name }_{ index }.xlsx' )

                best_model  = grid_search.best_estimator_
                best_params = grid_search.best_params_

                y_pred_train_class = best_model.predict( x_train )
                y_pred_train_proba = best_model.predict_proba( x_train )[ :, 1 ]            

                y_pred_test_class  = best_model.predict( x_test )
                y_pred_test_proba  = best_model.predict_proba( x_test )[ :, 1 ]
                
                joblib.dump( best_model, f'{ path_list[ 0 ] }/model_{ model_name }_{ index }.joblib' )

                if hasattr( best_model, 'feature_importances_' ):
                    
                    feature_importances = best_model.feature_importances_
                    vars_df             = pd.DataFrame( {'Var': pred_vars, 'Importance Score': feature_importances } )
                    vars_df             = vars_df.reindex( vars_df[ 'Importance Score' ].abs().sort_values( ascending = False ).index )
                    vars_df.to_excel( f'{ path_list[ 1 ] }/varlist_{ model_name }_{ index }.xlsx' )

                elif hasattr( best_model, 'coef_' ):
                    
                    coefficients = best_model.coef_[ 0 ]
                    vars_df      = pd.DataFrame( {'Var': best_model.feature_names_in_, 'Coefficient': coefficients } )
                    vars_df      = vars_df.reindex( vars_df[ 'Coefficient' ].abs().sort_values( ascending = False ).index )
                    vars_df.to_excel( f'{ path_list[ 1 ] }/varlist_{ model_name }_{ index }.xlsx' )

            else:
                model.fit( x_train, y_train )

                best_params  = 'No grid search'

                y_pred_train_class = model.predict( x_train )
                y_pred_train_proba = model.predict_proba( x_train )[ :, 1 ]            

                y_pred_test_class  = model.predict( x_test )
                y_pred_test_proba  = model.predict_proba( x_test )[ :, 1 ]
                
                joblib.dump( model, f'{ path_models }/{ model_name }_{ index }.joblib' )

                coefficients = model.coef_[ 0 ]
                vars_df      = pd.DataFrame( {'Var': model.feature_names_in_, 'Coefficient': coefficients } )
                vars_df      = vars_df.reindex( vars_df[ 'Coefficient' ].abs().sort_values( ascending = False ).index )
                vars_df.to_excel( f'{ path_list[ 1 ] }/varlist_{ model_name }_{ index }.xlsx' )

            accuracy_train  = accuracy_score( y_train, y_pred_train_class )
            log_loss_train  = log_loss( y_train, y_pred_train_class )
            roc_auc_train   = roc_auc_score( y_train, y_pred_train_proba )
            f1_score_train  = f1_score( y_train, y_pred_train_class, average = 'macro' )
            mcc_score_train = matthews_corrcoef( y_train, y_pred_train_class )

            accuracy_test   = accuracy_score( y_test, y_pred_test_class )
            log_loss_test   = log_loss( y_test, y_pred_test_class )
            roc_auc_test    = roc_auc_score( y_test, y_pred_test_proba )
            f1_score_test   = f1_score( y_test, y_pred_test_class, average = 'macro' )
            mcc_score_test  = matthews_corrcoef( y_test, y_pred_test_class )

            results[ 'Model' ].append( f'{ model_name }_{ index }' )
            results[ 'accuracy_train' ].append( round( accuracy_train, 3 ) )
            results[ 'accuracy_test' ].append( round( accuracy_test, 3 ) )
            results[ 'log_loss_train' ].append( round( log_loss_train, 3 ) )
            results[ 'log_loss_test' ].append( round( log_loss_test, 3 ) )
            results[ 'roc_auc_train' ].append( round( roc_auc_train, 3 ) )
            results[ 'roc_auc_test' ].append( round( roc_auc_test, 3 ) )
            results[ 'f1_train' ].append( round( f1_score_train, 3 ) )
            results[ 'f1_test' ].append( round( f1_score_test, 3 ) )
            results[ 'MCC_train' ].append( round( mcc_score_train, 3 ) )
            results[ 'MCC_test' ].append( round( mcc_score_test, 3 ) )          
            results[ 'Grid_Search_Params' ].append( best_params )
        
    results_df = pd.DataFrame( results )
    results_df = results_df.sort_values( by = 'f1_test', ascending = False )

    return results_df