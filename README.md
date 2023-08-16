# CORRUPTION PAPER - EMPIRICAL ANALYSIS

## 1. Información General

El presente proyecto utiliza tres bases de datos que fueron obtenidas de la siguiente manera: i) la base renamu se construyó a partir de los cinco módulos que presenta la encuesta, eliminando únicamente aquellas variables no consideradas relevantes para el proyecto, ii) la base siaf es una combinación de las funciones agrupadas por gastos corrientes, gastaos de capital y total de gastos para cada distrito, y iii) la base sentencias fue generada realizando web scraping a los resúmenes de las sentencias obtenidas de la contraloría.

* Primer procesamiento

A partir de la base sentencias se creó una nueva base panel en base a la diferencia entre el año de inicio y el año final del proceso, para posteriormente construir cuatro casos según la unidad la unidad de análisis.

(1) El primer caso, se construye a partir de la nueva bases sentencias donde se considera únicamente a la primera observación de cada reporte. Una vez se obtengan las observaciones a nivel de reporte se realiza un colapse por ubigeo y año de cada reporte de sentencia, este caso es conocido como caso inicial debido a que fue colapsado en base al año inicial de cada reporte de sentencia únicamente por ubigeo y año. [variables: 17 / observaciones: 657].

(2) En el segundo caso, se construye a partir de la nueva bases sentencias donde se considera únicamente a la primera observación de cada reporte donde será colapsada por ubigeo, año inicial y reporte de sentencia. Este caso se encuentra en función del primer año de cada reporte por ubigeo reporte y año [variables: 19 / observaciones: 850]. 

(3) Para el tercer caso se tiene a la contraloría panel que toma las diferencias entre el primer año y el último año de sentencia de acuerdo al reporte colapsada por ubigeo y año de diferencias del reporte [variables: 17 / observaciones: 1297]. 

(4) Por último, el cuarto caso toma a la contraloría panel que toma las diferencias entre el primer año y el último año de sentencia de acuerdo al reporte colapsada por ubigeo, año y caso del reporte de sentencia, donde además se incluye a la variable prueba que determina si una observación es AC (Auditoria de cumplimiento) o SCEHPI (Servicio de Control Específico a Hechos de Presunta Irregularidad) [variables: 19 / observaciones: 1976].

* Segundo procesamiento

A partir de limpieza realizada a la base sentencias se procedió a clasificar la corrupción para el caso peruano en tres categorías según su intensidad. Esto se basa en el análisis de (Avis et. al, 2019 : 1924) el cual clasifica las irregularidades de la base de datos usada en Brollo (2013) en tres categorías (i) mal manejo, (ii) corrupción moderada, y (iii) corrupción severa. Por lo que, para determinar los parámetros de los niveles de corrupción en el Perú la clasificación de las dos nuevas variables dummy fueron consultadas con abogados, donde se obtuvo que los reportes de sentencias se considerarán como (1) corrupción intensa siempre y cuando las sentencias indiquen que por lo menos a una persona cometió un delito penal; y (2) corrupción amplia siempre y cuando las sentencias indiquen que por lo menos a una persona cometió un delito civil o uno penal. De esta manera, se obtiene que la gravedad de los delitos cumple la siguiente regla de gravedad: administrativos < civil < penal. Por otro lado, para generar la variable monto se utilizaron las dos variables existentes en los reportes de las sentencias, se cumple por tanto que monto es igual a la suma del monto auditado presente y el monto objeto de servicio presente.
A partir de las variables dummys creadas se optó por generar cuatro variables interactivas que multiplican los valores de las variables numéricas y los valores de las categorías de las variables dummys: (1) número de personas que cometieron corrupción del primer nivel, (2) número de personas que cometieron corrupción del segundo nivel, (3) monto total de sentencias que se encuentran en el primer nivel de corrupción y (4) monto total de sentencias que se encuentran en el segundo nivel de corrupción.

* Tercer procesamiento

El proyecto combina tres bases de datos (renamu, siaf y sentencias) que fueron procesadas arbitrariamente para el presente proyecto.

Se realiza la unión de la base renamu y siaf (renamu_siaf), para finalizar la construcción de las bases finales para cada uno de los cuatro casos generados (1) contraloría inicial por ubigeo y año, (2) contraloria inicial por ubigeo, año y caso, (3) contraloria panel por ubigeo y año, y (4) contraloria panel por ubigeo, año y caso.


## 2. Procesamiento en Python

1.- Cargar los datos

1.1. Importar librerias y módulos

1.2. Importar bases

1.3. Filtrar solo aquellas variables que hicieron match

1.4. Última limpieza a las bases


## 3.	Bibliografía

Avis, Eric; Ferraz, C; Finan, F; (2018) “Do Govermente audits reduce corruption? Estimating the impacts of exposing corrupt Politicians”. Journal of Political Economy

Brollo, Fernanda (2013) “The Political Resource Curse” American Economic Review. Vol. 103 N-5

INEI (2020) “Estimación de la vulnerabilidad económica a la pobreza monetaria”

Ferraz, C., & Finan, F. (2008). Exposing corrupt politicians: the effects of Brazil's publicly released audits on electoral outcomes. The Quarterly journal of economics, 123(2), 703-745.

Olken, B. A., & Pande, R. (2012). Corruption in developing countries. Annu. Rev. Econ., 4(1), 479-509.
