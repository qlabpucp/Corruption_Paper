/********************************************************************************
			PROYECTO CORRUPCIÓN
********************************************************************************/

clear all
cls

set maxvar 120000

if "`c(username)'" == "dell" global main "C:\Users\dell\Documents\GitHub\Corruption_Paper"

if "`c(username)'" == "Usuario" global main "C:\Users\Usuario\Desktop\QLAB\GitHub\Corruption_Paper"

global input "$main\input\input_data"
global data "$main\input\built_data"
global extra "$main\extra"
global varnames "$main\extra\varnames"
global iecodebook "$main\extra\iecodebook"

*==========================================================================
* BASE CONTRALORIA: CONSTRUCCIÓN DE CUATRO CASOS DE VARIABLES DEPENDIENTES
*==========================================================================

* (1) contraloría del año inicial del reporte por ubigeo y año
*--------------------------------------------------------------

use $input\contraloria_caso, clear

rename año_inicio year
rename num_informe doc_name

* Agregar variable gasto
merge 1:1 doc_name using $input\matrix_variable_gasto
drop if _merge != 3
drop _merge

collapse (sum) civil penal admin adm_ent adm_pas monto_auditado monto_examinado monto_objeto_servicio (first) variable_gasto, by (ubigeo year tipo_control)
/*  variables: 12
observaciones: 2183  */

duplicates list ubigeo year
*214 obs duplicadas -> convertimos a todas en sepi
*keep ubigeo year tipo_control
*export excel using "C:\Users\Usuario\Downloads\aaaa.xls", firstrow(variables)
*keep ubigeo year
*duplicates tag, generate(dup)
*list if dup==1
*codebook dup
order ubigeo year penal civil admin adm_ent adm_pas monto_auditado monto_examinado monto_objeto_servicio variable_gasto

* variable "corrupción intensa"
gen corrup_intensa = 1 if penal != 0
replace corrup_intensa = 0 if corrup_intensa == .
tostring corrup_intensa, gen(str_corrup_intensa) force
encode str_corrup_intensa, gen(id_corrup_intensa)
drop corrup_intensa str_corrup_intensa
rename id_corrup_intensa corrup_intensa
label var corrup_intensa "CORRUPCIÓN INTENSA"
replace corrup_intensa = 0 if corrup_intensa == 1
replace corrup_intensa = 1 if corrup_intensa == 2
label define frec1 1 "Sí" 0 "No"
label list frec1
label values corrup_intensa frec1
codebook corrup_intensa

* variable "corrupción amplia"
gen corrup_amplia = 1 if civil != 0
replace corrup_amplia = 1 if penal != 0
replace corrup_amplia = 0 if corrup_amplia == .
tostring corrup_amplia, gen(str_corrup_amplia) force
encode str_corrup_amplia, gen(id_corrup_amplia)
drop corrup_amplia str_corrup_amplia
rename id_corrup_amplia corrup_amplia
label var corrup_amplia "CORRUPCIÓN AMPLIA"
replace corrup_amplia = 0 if corrup_amplia == 1
replace corrup_amplia = 1 if corrup_amplia == 2
label list frec1
label values corrup_amplia frec1
codebook corrup_amplia

* variable "monto_"
gen monto_ = monto_auditado + monto_objeto_servicio
label var monto_ "SUMA DE MONTOS: AUDITADO Y OBJETO"

* variable interactiva: monto auditado según prueba
gen monto = monto_auditado if tipo_control == 1
replace monto = monto_objeto_servicio if tipo_control == 2
label var monto "MONTO AUDITADO SEGÚN PRUEBA"

* variables interactivas: número de personas según tipo de corrupción
gen per_corrup1 = penal * corrup_intensa
gen per_corrup2 = civil * corrup_amplia
label var per_corrup1 "NÚMERO DE PERSONAS SEGÚN CORRUPCIÓN INTENSA"
label var per_corrup2 "NÚMERO DE PERSONAS SEGÚN CORRUPCIÓN AMPLIA"

* variables interactivas monto según tipo de corrupción
gen monto_corrup1 = monto * corrup_intensa
gen monto_corrup2 = monto * corrup_amplia
label var monto_corrup1 "MONTO SEGÚN CORRUPCIÓN INTENSA"
label var monto_corrup2 "MONTO SEGÚN CORRUPCIÓN AMPLIA"

* Variable gasto
label var variable_gasto "VARIABLE TIPO DE GASTO"
label define vg 1 "Gasto corriente" 2 "Gasto capital" 3 "Ambos" 4 "Indeterminado"
label values variable_gasto vg 


save $data/c1, replace
/*  variables: 20
observaciones: 2,183  */


* (2) contraloría del año inicial del reporte por ubigeo, caso y año
*--------------------------------------------------------------------

use $input\contraloria_caso, clear

rename año_inicio year
rename num_informe doc_name

* Agregar variable gasto
merge 1:1 doc_name using $input\matrix_variable_gasto
drop if _merge != 3
drop _merge

drop titulo_asunto objetivo entidad_auditada año_emision unidad_emite año_fin
order ubigeo year doc_name tipo_control penal civil admin adm_ent adm_pas monto_auditado monto_examinado monto_objeto_servicio variable_gasto
/*  variables: 13
observaciones: 3174  */

* variable "corrupción intensa"
gen corrup_intensa = 1 if penal != 0
replace corrup_intensa = 0 if corrup_intensa == .
tostring corrup_intensa, gen(str_corrup_intensa) force
encode str_corrup_intensa, gen(id_corrup_intensa)
drop corrup_intensa str_corrup_intensa
rename id_corrup_intensa corrup_intensa
label var corrup_intensa "CORRUPCIÓN INTENSA"
replace corrup_intensa = 0 if corrup_intensa == 1
replace corrup_intensa = 1 if corrup_intensa == 2
label define frec1 1 "Sí" 0 "No"
label list frec1
label values corrup_intensa frec1
codebook corrup_intensa

* variable "corrupción amplia"
gen corrup_amplia = 1 if civil != 0
replace corrup_amplia = 1 if penal != 0
replace corrup_amplia = 0 if corrup_amplia == .
tostring corrup_amplia, gen(str_corrup_amplia) force
encode str_corrup_amplia, gen(id_corrup_amplia)
drop corrup_amplia str_corrup_amplia
rename id_corrup_amplia corrup_amplia
label var corrup_amplia "CORRUPCIÓN AMPLIA"
replace corrup_amplia = 0 if corrup_amplia == 1
replace corrup_amplia = 1 if corrup_amplia == 2
label list frec1
label values corrup_amplia frec1
codebook corrup_amplia

* variable "monto_"
gen monto_ = monto_auditado + monto_objeto_servicio
label var monto_ "SUMA DE MONTOS: AUDITADO Y OBJETO"

* variable interactiva: monto auditado según prueba
gen monto = monto_auditado if tipo_control == 1
replace monto = monto_objeto_servicio if tipo_control == 2
label var monto "MONTO AUDITADO SEGÚN PRUEBA"

* variables interactivas: número de personas según tipo de corrupción
gen per_corrup1 = penal * corrup_intensa
gen per_corrup2 = civil * corrup_amplia
label var per_corrup1 "NÚMERO DE PERSONAS SEGÚN CORRUPCIÓN INTENSA"
label var per_corrup2 "NÚMERO DE PERSONAS SEGÚN CORRUPCIÓN AMPLIA"

* variables interactivas monto según tipo de corrupción
gen monto_corrup1 = monto * corrup_intensa
gen monto_corrup2 = monto * corrup_amplia
label var monto_corrup1 "MONTO SEGÚN CORRUPCIÓN INTENSA"
label var monto_corrup2 "MONTO SEGÚN CORRUPCIÓN AMPLIA"

* Variable gasto
label var variable_gasto "VARIABLE TIPO DE GASTO"
label define vg 1 "Gasto corriente" 2 "Gasto capital" 3 "Ambos" 4 "Indeterminado"
label values variable_gasto vg 

save $data/c2, replace
/*  variables: 21
observaciones: 3,174  */


* (3) contraloría panel por ubigeo y año
*----------------------------------------

use $input\contraloria_panel, clear
rename año year
rename num_informe doc_name

* Agregar variable gasto
merge m:1 doc_name using $input\matrix_variable_gasto
drop if _merge != 3
drop _merge

collapse (sum) civil penal admin adm_ent adm_pas monto_auditado monto_examinado monto_objeto_servicio (first) variable_gasto, by (ubigeo year tipo_control)
/*  variables: 11
observaciones: 4392  */

duplicates list ubigeo year
*694 obs duplicadas -> convertimos a todas en sepi
*keep ubigeo year tipo_control
*export excel using "C:\Users\Usuario\Downloads\bbbb.xls", firstrow(variables)
*keep ubigeo year
*duplicates tag, generate(dup)
*list if dup==1
*codebook dup
order ubigeo year penal civil admin adm_ent adm_pas monto_auditado monto_examinado monto_objeto_servicio variable_gasto

* variable "corrupción intensa"
gen corrup_intensa = 1 if penal != 0
replace corrup_intensa = 0 if corrup_intensa == .
tostring corrup_intensa, gen(str_corrup_intensa) force
encode str_corrup_intensa, gen(id_corrup_intensa)
drop corrup_intensa str_corrup_intensa
rename id_corrup_intensa corrup_intensa
label var corrup_intensa "CORRUPCIÓN INTENSA"
replace corrup_intensa = 0 if corrup_intensa == 1
replace corrup_intensa = 1 if corrup_intensa == 2
label define frec1 1 "Sí" 0 "No"
label list frec1
label values corrup_intensa frec1
codebook corrup_intensa

* variable "corrupción amplia"
gen corrup_amplia = 1 if civil != 0
replace corrup_amplia = 1 if penal != 0
replace corrup_amplia = 0 if corrup_amplia == .
tostring corrup_amplia, gen(str_corrup_amplia) force
encode str_corrup_amplia, gen(id_corrup_amplia)
drop corrup_amplia str_corrup_amplia
rename id_corrup_amplia corrup_amplia
label var corrup_amplia "CORRUPCIÓN AMPLIA"
replace corrup_amplia = 0 if corrup_amplia == 1
replace corrup_amplia = 1 if corrup_amplia == 2
label list frec1
label values corrup_amplia frec1
codebook corrup_amplia

* variable "monto_"
gen monto_ = monto_auditado + monto_objeto_servicio
label var monto_ "SUMA DE MONTOS: AUDITADO Y OBJETO"

* variable interactiva: monto auditado según prueba
gen monto = monto_auditado if tipo_control == 1
replace monto = monto_objeto_servicio if tipo_control == 2
label var monto "MONTO AUDITADO SEGÚN PRUEBA"

* variables interactivas: número de personas según tipo de corrupción
gen per_corrup1 = penal * corrup_intensa
gen per_corrup2 = civil * corrup_amplia
label var per_corrup1 "NÚMERO DE PERSONAS SEGÚN CORRUPCIÓN INTENSA"
label var per_corrup2 "NÚMERO DE PERSONAS SEGÚN CORRUPCIÓN AMPLIA"

* variables interactivas monto según tipo de corrupción
gen monto_corrup1 = monto * corrup_intensa
gen monto_corrup2 = monto * corrup_amplia
label var monto_corrup1 "MONTO SEGÚN CORRUPCIÓN INTENSA"
label var monto_corrup2 "MONTO SEGÚN CORRUPCIÓN AMPLIA"

* Variable gasto
label var variable_gasto "VARIABLE TIPO DE GASTO"
label define vg 1 "Gasto corriente" 2 "Gasto capital" 3 "Ambos" 4 "Indeterminado"
label values variable_gasto vg 

save $data/c3, replace
/*  variables: 20
observaciones: 4,392  */


* (4) contraloria panel por ubigeo, caso y año
*----------------------------------------------

use $input\contraloria_panel, clear
rename año year
rename num_informe doc_name

* Agregar variable gasto
merge m:1 doc_name using $input\matrix_variable_gasto
drop if _merge != 3
drop _merge

gen monto_auditado_promedio = monto_auditado / dif
gen monto_examinado_promedio = monto_examinado / dif
gen monto_objeto_promedio = monto_objeto_servicio / dif
drop monto_auditado monto_examinado monto_objeto_servicio

keep ubigeo year doc_name tipo_control penal civil admin adm_ent adm_pas monto_auditado_promedio monto_examinado_promedio monto_objeto_promedio variable_gasto
/*  variables: 12
observaciones: 1976  */
order ubigeo year doc_name penal civil admin adm_ent adm_pas monto_auditado_promedio monto_examinado_promedio monto_objeto_promedio variable_gasto

* variable "corrupción intensa"
gen corrup_intensa = 1 if penal != 0
replace corrup_intensa = 0 if corrup_intensa == .
tostring corrup_intensa, gen(str_corrup_intensa) force
encode str_corrup_intensa, gen(id_corrup_intensa)
drop corrup_intensa str_corrup_intensa
rename id_corrup_intensa corrup_intensa
label var corrup_intensa "CORRUPCIÓN INTENSA"
replace corrup_intensa = 0 if corrup_intensa == 1
replace corrup_intensa = 1 if corrup_intensa == 2
label define frec1 1 "Sí" 0 "No"
label list frec1
label values corrup_intensa frec1
codebook corrup_intensa

* variable "corrupción amplia"
gen corrup_amplia = 1 if civil != 0
replace corrup_amplia = 1 if penal != 0
replace corrup_amplia = 0 if corrup_amplia == .
tostring corrup_amplia, gen(str_corrup_amplia) force
encode str_corrup_amplia, gen(id_corrup_amplia)
drop corrup_amplia str_corrup_amplia
rename id_corrup_amplia corrup_amplia
label var corrup_amplia "CORRUPCIÓN AMPLIA"
replace corrup_amplia = 0 if corrup_amplia == 1
replace corrup_amplia = 1 if corrup_amplia == 2
label list frec1
label values corrup_amplia frec1
codebook corrup_amplia

* variable "monto_"
gen monto_ = monto_auditado_promedio + monto_objeto_promedio
label var monto_ "SUMA DE MONTOS: AUDITADO Y OBJETO"

* variable interactiva: monto auditado según prueba
gen monto = monto_auditado_promedio if tipo_control == 1
replace monto = monto_objeto_promedio if tipo_control == 2
label var monto "MONTO AUDITADO SEGÚN PRUEBA"

* variables interactivas: número de personas según tipo de corrupción
gen per_corrup1 = penal * corrup_intensa
gen per_corrup2 = civil * corrup_amplia
label var per_corrup1 "NÚMERO DE PERSONAS SEGÚN CORRUPCIÓN INTENSA"
label var per_corrup2 "NÚMERO DE PERSONAS SEGÚN CORRUPCIÓN AMPLIA"

* variables interactivas monto según tipo de corrupción
gen monto_corrup1 = monto * corrup_intensa
gen monto_corrup2 = monto * corrup_amplia
label var monto_corrup1 "MONTO SEGÚN CORRUPCIÓN INTENSA"
label var monto_corrup2 "MONTO SEGÚN CORRUPCIÓN AMPLIA"

* Variable gasto
label var variable_gasto "VARIABLE TIPO DE GASTO"
label define vg 1 "Gasto corriente" 2 "Gasto capital" 3 "Ambos" 4 "Indeterminado"
label values variable_gasto vg 

save $data/c4, replace
/*  variables: 20
observaciones: 1,976 */


********************************************************************************
* UNIÓN DE LOS CUATRO CASOS CON LAS BASES RENAMU - SIAF
********************************************************************************

* RENAMU - SIAF - POLITICA
use $input\matrix_renamu, clear
rename idmunici ubigeo
merge 1:1 ubigeo year using $input\matrix_siaf
drop _merge
save $input/matrix_renamu_siaf, replace
use $input\matrix_renamu_siaf, clear
merge 1:1 ubigeo year using $input\matrix_politica
drop _merge
save $input\matrix_renamu_siaf_politica, replace
/*  variables: 18,283
observaciones: 25,915  */


* (1) contraloría del año inicial del reporte por ubigeo y año
*--------------------------------------------------------------
use $input/matrix_renamu_siaf_politica, clear
merge 1:m ubigeo year using $data/c1
/*  variables: 18,301
observaciones: 26,249  */
drop if _merge != 3
drop _merge
drop if sharevotprim == . // drop if not present in political vars
save $data/matrix_c1, replace
/*  variables: 18,301
observaciones: 1,794  */


* (2) contraloría del año inicial del reporte por ubigeo, caso y año
*--------------------------------------------------------------------
use $input/matrix_renamu_siaf_politica, clear
merge 1:m ubigeo year using $data/c2
/*  variables: 18,302
observaciones: 27,241  */
drop if _merge != 3
drop _merge
drop if sharevotprim == . // drop if not present in political vars
save $data/matrix_c2, replace
/*  variables: 18,302
observaciones: 2,700  */


* (3) contraloría panel por ubigeo y año
*----------------------------------------
use $input/matrix_renamu_siaf_politica, clear
merge 1:m ubigeo year using $data/c3
/*  variables: 18,301
observaciones: 27,070  */
drop if _merge != 3
drop _merge
drop if sharevotprim == . // drop if not present in political vars
save $data/matrix_c3, replace
/*  variables: 18,301
observaciones: 3,370  */


* (4) contraloria panel por ubigeo, caso y año
*----------------------------------------------
use $input/matrix_renamu_siaf_politica, clear
merge 1:m ubigeo year using $data/c4
/*  variables: 18,302
observaciones: 30,365  */
drop if _merge != 3
drop _merge
drop if sharevotprim == . // drop if not present in political vars
save $data/matrix_c4, replace
/*  variables: 18,302
observaciones: 6,269  */

/*
erase $input/matrix_renamu_siaf_politica.dta
erase $data/c1.dta
erase $data/c2.dta
erase $data/c3.dta
erase $data/c4.dta
*/


********************************************************************************
* EXPORTAR LISTAS DE VARIABLES DE CADA UNA DE LAS BASES
********************************************************************************

* RENAMU
use "$input/matrix_renamu", clear
describe, replace
export excel name varlab using "$varnames/renamu_variables.xlsx", firstrow(variables)

* SIAF FINAL
use "$input/matrix_siaf", clear
describe, replace
export excel name varlab using "$varnames/siaf_variables.xlsx", firstrow(variables)

* CONTRALORIA
use "$input/matrix_contraloria", clear
describe, replace
export excel name varlab using "$varnames/contraloria_variables.xlsx", firstrow(variables)


********************************************************************************
* IETOOLKIT
********************************************************************************
// ssc install iefieldkit

* (1) contraloría del año inicial del reporte por ubigeo y año
*--------------------------------------------------------------
use $data/matrix_c1, clear
iecodebook template using "$iecodebook\cleaning_c1.xlsx", replace
iecodebook apply using "$iecodebook\cleaning_c1.xlsx"

* (2) contraloría del año inicial del reporte por ubigeo, caso y año
*--------------------------------------------------------------------
use $data/matrix_c2, clear
iecodebook template using "$iecodebook\cleaning_c2.xlsx", replace
iecodebook apply using "$iecodebook\cleaning_c2.xlsx"

* (3) contraloría panel por ubigeo y año
*----------------------------------------
use $data/matrix_c3, clear
iecodebook template using "$iecodebook\cleaning_c3.xlsx", replace
iecodebook apply using "$iecodebook\cleaning_c3.xlsx"

* (4) contraloria panel por ubigeo, caso y año
*----------------------------------------------
use $data/matrix_c4, clear
iecodebook template using "$iecodebook\cleaning_c4.xlsx", replace
iecodebook apply using "$iecodebook\cleaning_c4.xlsx"
