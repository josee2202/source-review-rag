"""Corpus SINTÉTICO para la demo y la evaluación.

Todo el contenido es ficticio y se ha redactado para este repositorio: "Banco Ficticio S.A."
no existe y la "guía supervisora" no reproduce ninguna norma real (se inspira en principios
habituales de la regulación prudencial y contable sobre riesgo de crédito).

El documento metodológico incluye desviaciones deliberadas respecto a la guía. Cada sección
lleva su etiqueta de referencia (`gold_req`, `gold_label`) para evaluar la recuperación y el
veredicto del LLM. Las secciones con `gold_req=None` no tienen requisito aplicable en la guía.
"""

TITULO_NORMATIVA = "Guía supervisora sintética sobre la estimación de la PD"
INTRO_NORMATIVA = (
    "Documento ficticio redactado exclusivamente para demostrar el sistema. No reproduce ninguna "
    "norma real, aunque se inspira en principios habituales de la regulación prudencial y contable "
    "sobre riesgo de crédito."
)

REQUISITOS = {
    "R1": ("Definición de default",
           "La entidad considerará que una exposición está en situación de default cuando el deudor "
           "acumule más de 90 días de impago en una obligación significativa o cuando se considere "
           "improbable que pague sus obligaciones sin recurrir a la ejecución de garantías. La definición "
           "se aplicará de forma homogénea en todas las carteras, en la estimación de parámetros y en la "
           "gestión diaria del riesgo."),
    "R2": ("Profundidad histórica de los datos",
           "La tasa media de default a largo plazo utilizada para calibrar la PD se calculará con un "
           "periodo de observación de al menos cinco años. Si la entidad dispone de un histórico más "
           "largo y relevante, deberá utilizarlo. Las exclusiones de periodos deberán justificarse y "
           "documentarse."),
    "R3": ("Ciclo económico y margen de conservadurismo",
           "El periodo de observación deberá incluir una combinación representativa de años favorables "
           "y desfavorables del ciclo económico. Cuando los datos no sean representativos o presenten "
           "deficiencias, la entidad aplicará un margen de conservadurismo cuantificado, documentado y "
           "proporcional a la incertidumbre de la estimación."),
    "R4": ("Calibración y backtesting",
           "Al menos una vez al año, la entidad comparará la PD estimada de cada grado de rating con la "
           "tasa de default observada, utilizando el test binomial u otra prueba estadística "
           "equivalente. Las desviaciones significativas deberán analizarse y dar lugar a acciones "
           "correctoras documentadas."),
    "R5": ("Poder discriminante",
           "La capacidad del modelo para ordenar el riesgo se medirá al menos anualmente mediante el AUC "
           "o el coeficiente de Gini. La entidad fijará umbrales mínimos de rendimiento y un "
           "procedimiento de escalado cuando dichos umbrales no se alcancen."),
    "R6": ("Calidad de los datos",
           "Los datos utilizados en el desarrollo y la aplicación del modelo deberán ser completos, "
           "exactos y trazables hasta los sistemas de origen. La entidad realizará conciliaciones "
           "periódicas con la información contable y documentará los controles aplicados."),
    "R7": ("Ajustes manuales",
           "Los ajustes manuales sobre las calificaciones (overrides) deberán estar justificados, "
           "documentados y limitados en volumen. Una función independiente de la unidad de desarrollo "
           "revisará periódicamente su uso y su impacto en los parámetros."),
    "R8": ("Información prospectiva",
           "Para la estimación de pérdidas esperadas bajo IFRS 9, la PD incorporará información "
           "prospectiva mediante al menos tres escenarios macroeconómicos (base, adverso y favorable) "
           "ponderados por su probabilidad de ocurrencia. La elección de escenarios y ponderaciones "
           "deberá justificarse."),
    "R9": ("PD a lo largo de la vida de la operación",
           "Para las exposiciones con incremento significativo del riesgo, la entidad estimará la PD "
           "durante toda la vida esperada de la operación, utilizando curvas de supervivencia, matrices "
           "de transición u otra metodología debidamente justificada."),
    "R10": ("Validación independiente",
            "Una unidad de validación independiente de la de desarrollo revisará el modelo al menos una "
            "vez al año. Sus conclusiones y recomendaciones se comunicarán a la alta dirección y se "
            "realizará un seguimiento de su implantación."),
}

TITULO_METODOLOGIA = "Documento metodológico del modelo de PD de hipotecas - Banco Ficticio S.A."
INTRO_METODOLOGIA = "Documento ficticio redactado para demostrar el sistema. Banco Ficticio S.A. no existe."

# gold_label: "alineado" | "no_alineado" | "sin_evidencia"
METODOLOGIA = [
    {"id": "M1", "titulo": "Definición de default", "gold_req": "R1", "gold_label": "alineado",
     "texto": "El modelo considera en default a los préstamos hipotecarios con más de 90 días de impago en "
              "un importe significativo, así como a las operaciones en las que se identifica improbabilidad "
              "de pago, por ejemplo por concurso del deudor o refinanciación forzosa. La misma definición se "
              "utiliza en el desarrollo del modelo y en la gestión diaria del riesgo de la cartera."},
    {"id": "M2", "titulo": "Muestra de desarrollo", "gold_req": "R2", "gold_label": "no_alineado",
     "texto": "La tasa media de default a largo plazo se ha calculado con la muestra disponible entre enero "
              "de 2021 y diciembre de 2023, es decir, tres años de observación. Los años anteriores se han "
              "descartado porque procedían de un sistema informático antiguo, sin que se haya documentado un "
              "análisis del impacto de esta exclusión."},
    {"id": "M3", "titulo": "Ciclo económico y conservadurismo", "gold_req": "R3", "gold_label": "no_alineado",
     "texto": "El periodo de observación corresponde a una fase expansiva del ciclo económico, con tasas de "
              "default históricamente bajas. La entidad no aplica ningún margen de conservadurismo sobre la "
              "PD calibrada, al considerar que el comportamiento reciente de la cartera es representativo "
              "del futuro."},
    {"id": "M4", "titulo": "Calibración", "gold_req": "R4", "gold_label": "alineado",
     "texto": "Cada año se compara la PD asignada a cada grado de rating con la tasa de default observada "
              "mediante el test binomial con un nivel de confianza del 95%. Las desviaciones significativas "
              "se analizan en el comité de modelos y se documentan junto con las acciones correctoras "
              "acordadas."},
    {"id": "M5", "titulo": "Poder discriminante", "gold_req": "R5", "gold_label": "alineado",
     "texto": "El poder discriminante se mide anualmente con el AUC, que en la última revisión fue de 0,81. "
              "Se ha fijado un umbral mínimo de 0,70; si el AUC queda por debajo, el caso se escala al "
              "comité de riesgos para decidir si se recalibra o se redesarrolla el modelo."},
    {"id": "M6", "titulo": "Escenarios prospectivos", "gold_req": "R8", "gold_label": "no_alineado",
     "texto": "Para el cálculo de la pérdida esperada bajo IFRS 9 se utiliza un único escenario "
              "macroeconómico, el escenario base publicado por el servicio de estudios, sin escenarios "
              "alternativos ni ponderaciones por probabilidad."},
    {"id": "M7", "titulo": "Calidad de los datos", "gold_req": "R6", "gold_label": "alineado",
     "texto": "Los datos del modelo proceden del almacén corporativo y se concilian mensualmente con la "
              "contabilidad. Cada variable es trazable hasta su sistema de origen y los controles de "
              "completitud y exactitud se documentan en un registro de calidad del dato."},
    {"id": "M8", "titulo": "Ajustes manuales", "gold_req": "R7", "gold_label": "alineado",
     "texto": "Los ajustes manuales de rating requieren una justificación escrita, quedan registrados en la "
              "herramienta de admisión y no pueden superar el 5% de las operaciones. La unidad de validación "
              "revisa su uso cada semestre."},
    {"id": "M9", "titulo": "Organización del equipo", "gold_req": None, "gold_label": "sin_evidencia",
     "texto": "El equipo de modelización está formado por cuatro analistas y un responsable. Las reuniones de "
              "seguimiento se celebran los lunes y la documentación de trabajo se guarda en la intranet del "
              "departamento."},
    {"id": "M10", "titulo": "Formato del documento", "gold_req": None, "gold_label": "sin_evidencia",
     "texto": "Este documento sigue la guía de estilo corporativa: tipografía Arial de 11 puntos, títulos "
              "numerados y una tabla de control de versiones en la portada."},
]

TITULO_TECNICO = "Documento técnico de implementación del modelo de PD de hipotecas - Banco Ficticio S.A."
INTRO_TECNICO = "Documento ficticio redactado para demostrar el sistema. Describe la implementación en SAS."

TECNICO = [
    ("Fuentes de datos",
     "Las operaciones se extraen mensualmente de las tablas del almacén corporativo. Antes de cada "
     "ejecución se concilian los saldos con la contabilidad y se registran los controles de completitud."),
    ("Construcción de la muestra",
     "La muestra de desarrollo incluye las operaciones hipotecarias observadas entre enero de 2020 y "
     "diciembre de 2023."),
    ("Marca de default",
     "La variable de default se activa cuando la operación supera los 90 días de impago o cuando está "
     "marcada con improbabilidad de pago."),
    ("Estimación del modelo",
     "La PD se estima con una regresión logística (PROC LOGISTIC) con selección stepwise sobre el LTV, la "
     "antigüedad de la operación y el ratio de esfuerzo. Después se asignan cinco grados de rating por "
     "tramos de PD."),
    ("Backtesting",
     "Para cada grado se calcula el límite superior del número de defaults con el test binomial al 95% y "
     "se marca el grado como rechazado si los defaults observados lo superan. El AUC se obtiene con la "
     "instrucción ROC de PROC LOGISTIC."),
    ("Ajustes manuales",
     "La herramienta de admisión permite ajustes manuales del rating hasta un máximo del 10% de las "
     "operaciones."),
    ("Escenarios macroeconómicos",
     "La PD prospectiva se calcula con tres escenarios (base, adverso y favorable) ponderados al 60%, "
     "25% y 15%."),
]

CODIGO_SAS = """/* Código SAS FICTICIO para la demo: modelo de PD de hipotecas de Banco Ficticio S.A. */
%let fecha_ini = '01JAN2020'd;
%let fecha_fin = '31DEC2023'd;

/* 1. Selección de la muestra de desarrollo */
data muestra;
    set dwh.operaciones_hipotecarias;
    where fecha_observacion between &fecha_ini. and &fecha_fin.;
run;

/* 2. Marca de default: más de 90 días de impago o improbabilidad de pago */
%macro marcar_default(tabla);
    data &tabla.;
        set &tabla.;
        default_flag = (dias_impago > 90 or flag_improbable_pago = 1);
    run;
%mend marcar_default;
%marcar_default(muestra);

/* 3. Estimación del modelo logístico */
%macro estimar_pd(tabla, salida);
    proc logistic data=&tabla. descending;
        model default_flag = ltv antiguedad ratio_esfuerzo / selection=stepwise;
        output out=&salida. p=pd_estimada;
    run;
%mend estimar_pd;
%estimar_pd(muestra, muestra_pd);

/* 4. Asignación de grados de rating por tramos de PD */
data muestra_pd;
    set muestra_pd;
    if pd_estimada < 0.005 then grado = 1;
    else if pd_estimada < 0.01 then grado = 2;
    else if pd_estimada < 0.03 then grado = 3;
    else if pd_estimada < 0.08 then grado = 4;
    else grado = 5;
run;

/* 5. Backtesting: test binomial por grado al 95% */
%macro test_binomial(tabla);
    proc sql;
        create table backtest as
        select grado,
               count(*) as n_operaciones,
               sum(default_flag) as n_defaults,
               mean(pd_estimada) as pd_media
        from &tabla.
        group by grado;
    quit;
    data backtest;
        set backtest;
        limite_sup = quantile('BINOMIAL', 0.95, pd_media, n_operaciones);
        rechazo = (n_defaults > limite_sup);
    run;
%mend test_binomial;
%test_binomial(muestra_pd);

/* 6. Poder discriminante: AUC */
proc logistic data=muestra_pd descending;
    model default_flag = pd_estimada / nofit;
    roc 'PD estimada' pred=pd_estimada;
run;
"""
