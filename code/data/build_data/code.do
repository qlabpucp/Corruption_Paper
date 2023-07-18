/********************************************************************************
			BASE CONTRALORÍA
********************************************************************************/

clear all
cls

set maxvar 10000

global main "C:\Users\dell\Documents\QLAB\Contraloria\base_construccion"
global works "$main\works"
cd "$main"

*====================
* BASES POR SEPARADO
*====================

* ubigeo
*--------
import excel "$main\concatenar_ubigeo", sheet("TB_UBIGEOS") firstrow
keep concatenar_ ubigeos_0_inei
save "$works\ubigeo.dta", replace

* base completa
*--------------
import excel "$main\concatenar_final", sheet("Sheet1") firstrow clear
merge m:1  concatenar_ using "$works\ubigeo.dta"
drop if _merge==2
drop A region provincia distrito observaciones recomendaciones argumentos_de_hecho filas concatenar_ _merge
rename (ubigeos_0_inei region_ provincia_ distrito_ unidad_emite_informe fecha_emision_informe titulo_informeasunto nombres_prueba inicio final)(ubigeo region provincia distrito unidad_emite fecha_emision titulo_asunto nom_apell fecha_inicio fecha_fin)
/*
replace civil = "0" if civil == "2"
replace penal = "0" if penal == "2"
replace admin = "0" if admin == "2"
replace adm_ent = "0" if adm_ent == "2"
replace adm_pas ="0" if adm_pas == "2" 
*/
destring civil penal admin adm_ent adm_pas monto_auditado monto_examinado monto_objeto_servicio, replace force
save "$works\final.dta", replace

use "$works/final.dta", clear
duplicates list doc_name dni
// Sale un reporte sin DNIs, que mantenemos

// drop if doc_name=="001-2021-OCI_1686-SOO-resumen.pdf"

*tipo de control
encode tipo_control, gen(id_tipo_control)
**#
drop tipo_control
rename id_tipo_control tipo_control
codebook tipo_control

*año - fecha de emisión
gen año_emision_=substr(fecha_emision,-6,4)
destring año_emision_, gen (año_emision)
drop fecha_emision año_emision_
label var año_emision "Año de emisión del informe"

*año - inicial y final
gen año_inicio_=substr(fecha_inicio,-4,4)
gen año_fin_=substr(fecha_fin,-4,4)
destring año_inicio_, gen (año_inicio)
destring año_fin_, gen (año_fin)
drop fecha_inicio fecha_fin año_inicio_ año_fin_
label var año_inicio "Año inicial del evento delictivo"
label var año_fin "Año final del evento delictivo"

drop region provincia distrito
order ubigeo doc_name titulo_asunto objetivo entidad_auditada monto_auditado monto_examinado monto_objeto_servicio año_emision unidad_emite tipo_control dni nom_apell civil penal admin adm_ent adm_pas año_inicio año_fin

label var ubigeo "Ubigeo"
label var doc_name "Número del informe"
label var titulo_asunto "Título del informe"
label var objetivo "Objetivo de la auditoría"
label var entidad_auditada "Entidad auditada"
label var monto_auditado "Monto auditado"
label var monto_examinado "Monto examinado"
label var monto_objeto_servicio "Monto objeto del servicio (por corresponder)"
label var unidad_emite "Unidad orgánica que emite el informe"
label var dni "Documento Nacional de Identidad (DNI)"
label var nom_apell "Personas identificadas en los hechos"
label var civil "Número de personas con responsabilidad civil"
label var penal "Número de personas con responsabilidad penal"
label var admin "Número de personas con responsabilidad administrativa"
label var adm_ent "Número de personas con responsabilidad administrativa  (Ent)"
label var adm_pas "Número de personas con responsabilidad administrativa  (Pas)"
label var tipo_control "Tipo de Control registrado en el resumen del informe"

save  matrix_contraloria.dta, replace


*===============================
* CREACIÓN DE DOS TIPOS DE BASE
*===============================

* (1) CONTRALORÍA - CASO
*------------------------

use matrix_contraloria.dta,clear
collapse (sum) civil penal admin adm_ent adm_pas, by (ubigeo doc_name titulo_asunto objetivo entidad_auditada monto_auditado monto_examinado monto_objeto_servicio año_emision unidad_emite tipo_control año_inicio año_fin)
duplicates list doc_name ubigeo
order ubigeo doc_name titulo_asunto objetivo entidad_auditada monto_auditado monto_examinado monto_objeto_servicio año_emision unidad_emite tipo_control civil penal admin adm_ent adm_pas año_inicio año_fin
save contraloria_caso.dta, replace

* (1) CONTRALORÍA - PANEL
*-------------------------

use contraloria_caso.dta,clear

*diferencia de años
gen dif=año_fin+1-año_inicio

* separo y uno bases por diferencia de años
*diferencia=1
preserve
keep if dif==1
gen año=año_inicio
save "$works\basedif1", replace
restore

*diferencia=2
preserve
keep if dif==2
save "$works\basedif2", replace
foreach var in año_inicio {
use "$works\basedif2", clear
gen `var'_=`var'
rename `var'_ año
save "$works\basedif21", replace
}
foreach var in año_fin {
use "$works\basedif2", clear
gen `var'_=`var'
rename `var'_ año
save "$works\basedif22", replace
}
use "$works\basedif21", clear
append using "$works\basedif22"
save "$works\basedif22", replace
restore

*diferencia=3
preserve
keep if dif==3
save "$works\basedif3", replace
foreach i in año_inicio {
use "$works\basedif3",clear
gen año=`i'
save "$works\basedif31",replace
use "$works\basedif3",clear
gen año=`i'+1
save "$works\basedif32",replace
use "$works\basedif3",clear
gen año=`i'+2
save "$works\basedif33",replace
}
forvalues i=31(1)32 {
use "$works\basedif`i'",clear
append using "$works\basedif`=`i'+1'"
save "$works\basedif`=`i'+1'", replace
}
restore

*diferencia=4
preserve
keep if dif==4
save "$works\basedif4", replace
foreach i in año_inicio {
use "$works\basedif4",clear
gen año=`i'
save "$works\basedif41",replace
use "$works\basedif4",clear
gen año=`i'+1
save "$works\basedif42",replace
use "$works\basedif4",clear
gen año=`i'+2
save "$works\basedif43",replace
use "$works\basedif4",clear
gen año=`i'+3
save "$works\basedif44",replace
}
forvalues i=41(1)43 {
use "$works\basedif`i'",clear
append using "$works\basedif`=`i'+1'"
save "$works\basedif`=`i'+1'", replace
}
restore

*diferencia=5
preserve
keep if dif==5
save "$works\basedif5", replace
foreach i in año_inicio {
use "$works\basedif5",clear
gen año=`i'
save "$works\basedif51",replace
use "$works\basedif5",clear
gen año=`i'+1
save "$works\basedif52",replace
use "$works\basedif5",clear
gen año=`i'+2
save "$works\basedif53",replace
use "$works\basedif5",clear
gen año=`i'+3
save "$works\basedif54",replace
use "$works\basedif5",clear
gen año=`i'+4
save "$works\basedif55",replace
}
forvalues i=51(1)54 {
use "$works\basedif`i'",clear
append using "$works\basedif`=`i'+1'"
save "$works\basedif`=`i'+1'", replace
}
restore

*diferencia=6
preserve
keep if dif==6
save "$works\basedif6", replace
foreach i in año_inicio {
use "$works\basedif6",clear
gen año=`i'
save "$works\basedif61",replace
use "$works\basedif6",clear
gen año=`i'+1
save "$works\basedif62",replace
use "$works\basedif6",clear
gen año=`i'+2
save "$works\basedif63",replace
use "$works\basedif6",clear
gen año=`i'+3
save "$works\basedif64",replace
use "$works\basedif6",clear
gen año=`i'+4
save "$works\basedif65",replace
use "$works\basedif6",clear
gen año=`i'+5
save "$works\basedif66",replace
}
forvalues i=61(1)65 {
use "$works\basedif`i'",clear
append using "$works\basedif`=`i'+1'"
save "$works\basedif`=`i'+1'", replace
}
restore

*diferencia=7
preserve
keep if dif==7
save "$works\basedif7", replace
foreach i in año_inicio {
use "$works\basedif7",clear
gen año=`i'
save "$works\basedif71",replace
use "$works\basedif7",clear
gen año=`i'+1
save "$works\basedif72",replace
use "$works\basedif7",clear
gen año=`i'+2
save "$works\basedif73",replace
use "$works\basedif7",clear
gen año=`i'+3
save "$works\basedif74",replace
use "$works\basedif7",clear
gen año=`i'+4
save "$works\basedif75",replace
use "$works\basedif7",clear
gen año=`i'+5
save "$works\basedif76",replace
use "$works\basedif7",clear
gen año=`i'+6
save "$works\basedif77", replace
}
forvalues i=71(1)76 {
use "$works\basedif`i'",clear
append using "$works\basedif`=`i'+1'"
save "$works\basedif`=`i'+1'", replace
}
restore

*diferencia=8
preserve
keep if dif==8
save "$works\basedif8", replace
foreach i in año_inicio {
use "$works\basedif8",clear
gen año=`i'
save "$works\basedif81",replace
use "$works\basedif8",clear
gen año=`i'+1
save "$works\basedif82",replace
use "$works\basedif8",clear
gen año=`i'+2
save "$works\basedif83",replace
use "$works\basedif8",clear
gen año=`i'+3
save "$works\basedif84",replace
use "$works\basedif8",clear
gen año=`i'+4
save "$works\basedif85",replace
use "$works\basedif8",clear
gen año=`i'+5
save "$works\basedif86",replace
use "$works\basedif8",clear
gen año=`i'+6
save "$works\basedif87", replace
use "$works\basedif8",clear
gen año=`i'+7
save "$works\basedif88", replace
}
forvalues i=81(1)87 {
use "$works\basedif`i'",clear
append using "$works\basedif`=`i'+1'"
save "$works\basedif`=`i'+1'", replace
}
restore

*diferencia=9
preserve
keep if dif==9
save "$works\basedif9", replace
foreach i in año_inicio {
use "$works\basedif9",clear
gen año=`i'
save "$works\basedif91",replace
use "$works\basedif9",clear
gen año=`i'+1
save "$works\basedif92",replace
use "$works\basedif9",clear
gen año=`i'+2
save "$works\basedif93",replace
use "$works\basedif9",clear
gen año=`i'+3
save "$works\basedif94",replace
use "$works\basedif9",clear
gen año=`i'+4
save "$works\basedif95",replace
use "$works\basedif9",clear
gen año=`i'+5
save "$works\basedif96",replace
use "$works\basedif9",clear
gen año=`i'+6
save "$works\basedif97", replace
use "$works\basedif9",clear
gen año=`i'+7
save "$works\basedif98", replace
use "$works\basedif9",clear
gen año=`i'+8
save "$works\basedif99", replace
}
forvalues i=91(1)98 {
use "$works\basedif`i'",clear
append using "$works\basedif`=`i'+1'"
save "$works\basedif`=`i'+1'", replace
}
restore

*diferencia=10
preserve
keep if dif==10
save "$works\basedif10", replace
foreach i in año_inicio {
use "$works\basedif10",clear
gen año=`i'
save "$works\basedif101",replace
use "$works\basedif10",clear
gen año=`i'+1
save "$works\basedif102",replace
use "$works\basedif10",clear
gen año=`i'+2
save "$works\basedif103",replace
use "$works\basedif10",clear
gen año=`i'+3
save "$works\basedif104",replace
use "$works\basedif10",clear
gen año=`i'+4
save "$works\basedif105",replace
use "$works\basedif10",clear
gen año=`i'+5
save "$works\basedif106",replace
use "$works\basedif10",clear
gen año=`i'+6
save "$works\basedif107", replace
use "$works\basedif10",clear
gen año=`i'+7
save "$works\basedif108", replace
use "$works\basedif10",clear
gen año=`i'+8
save "$works\basedif109", replace
use "$works\basedif10",clear
gen año=`i'+9
save "$works\basedif110", replace
}
forvalues i=101(1)109 {
use "$works\basedif`i'",clear
append using "$works\basedif`=`i'+1'"
save "$works\basedif`=`i'+1'", replace
}
restore

*diferencia=11
preserve
keep if dif==11
save "$works\basedif11", replace
foreach i in año_inicio {
use "$works\basedif11",clear
gen año=`i'
save "$works\basedif111",replace
use "$works\basedif11",clear
gen año=`i'+1
save "$works\basedif112",replace
use "$works\basedif11",clear
gen año=`i'+2
save "$works\basedif113",replace
use "$works\basedif11",clear
gen año=`i'+3
save "$works\basedif114",replace
use "$works\basedif11",clear
gen año=`i'+4
save "$works\basedif115",replace
use "$works\basedif11",clear
gen año=`i'+5
save "$works\basedif116",replace
use "$works\basedif11",clear
gen año=`i'+6
save "$works\basedif117", replace
use "$works\basedif11",clear
gen año=`i'+7
save "$works\basedif118", replace
use "$works\basedif11",clear
gen año=`i'+8
save "$works\basedif119", replace
use "$works\basedif11",clear
gen año=`i'+9
save "$works\basedif120", replace
use "$works\basedif11",clear
gen año=`i'+10
save "$works\basedif121", replace
}
forvalues i=111(1)120 {
use "$works\basedif`i'",clear
append using "$works\basedif`=`i'+1'"
save "$works\basedif`=`i'+1'", replace
}
restore

*diferencia=12
preserve
keep if dif==12
save "$works\basedif12", replace
foreach i in año_inicio {
use "$works\basedif12",clear
gen año=`i'
save "$works\basedif131",replace
use "$works\basedif12",clear
gen año=`i'+1
save "$works\basedif132",replace
use "$works\basedif12",clear
gen año=`i'+2
save "$works\basedif133",replace
use "$works\basedif12",clear
gen año=`i'+3
save "$works\basedif134",replace
use "$works\basedif12",clear
gen año=`i'+4
save "$works\basedif135",replace
use "$works\basedif12",clear
gen año=`i'+5
save "$works\basedif136",replace
use "$works\basedif12",clear
gen año=`i'+6
save "$works\basedif137", replace
use "$works\basedif12",clear
gen año=`i'+7
save "$works\basedif138", replace
use "$works\basedif12",clear
gen año=`i'+8
save "$works\basedif139", replace
use "$works\basedif12",clear
gen año=`i'+9
save "$works\basedif140", replace
use "$works\basedif12",clear
gen año=`i'+10
save "$works\basedif141", replace
use "$works\basedif12",clear
gen año=`i'+11
save "$works\basedif142", replace
}
forvalues i=131(1)141 {
use "$works\basedif`i'",clear
append using "$works\basedif`=`i'+1'"
save "$works\basedif`=`i'+1'", replace
}
restore

*uno todas las bases
use "$works\basedif1", clear
save "$works\_11", replace
forvalues i=22(11)121 {
use "$works\basedif`i'",clear
save "$works\_`i'", replace
}
use "$works\basedif142", clear
save "$works\_132", replace
forvalues i=11(11)121 {
use "$works\_`i'",clear
append using "$works\_`=`i'+11'"
save "$works\_`=`i'+11'", replace
}

duplicates list ubigeo doc_name año

order ubigeo doc_name titulo_asunto objetivo entidad_auditada monto_auditado monto_examinado monto_objeto_servicio año_emision unidad_emite tipo_control civil penal admin adm_ent adm_pas año_inicio año_fin

save contraloria_panel.dta, replace
