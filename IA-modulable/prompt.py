from langchain_core.prompts import PromptTemplate

# En la API, {contexto} son los fragmentos recuperados del documento de REFERENCIA (la norma)
# y {pregunta} es el fragmento del documento que se VALIDA (el del banco).
prompt_definitivos = {"evaluar_alineacion_normativa": PromptTemplate(
        input_variables=["contexto", "pregunta"],
        template='''
        Eres un experto en modelos de riesgo de crédito y análisis de fragmentos de texto sobre modelos de riesgo de crédito.

        **Objetivo:** Evaluar si el contexto proporcionado (sobre modelos de riesgo de crédito) se ajusta correctamente a las normativas y regulaciones
        aplicables (supervisor). Justifica si el contexto proporcionado está alineado con la norma y regulación del texto supervisor.

        **Información a analizar (contexto):**
        {pregunta}

        **Texto del supervisor (norma):**
        {contexto}

        **Salida:** Justifica de manera clara y concisa si el contexto se alinea con la normativa (supervisor). Responde EXCLUSIVAMENTE al contexto y texto del supervisor proporcionado,
        sin añadir interpretaciones fuera del análisis.
        La respuesta debe de ser proporcional al tamaño del contexto y del texto del supervisor, no añadas justificaciones fuera del análisis.
        '''
    ),   
      "evaluar_tecnico_funcional": PromptTemplate(
        input_variables=["contexto", "pregunta"],
        template="""
    Eres un experto en documentación técnica y metodológica del secor bancario. Valida que la explicación de la implementación (consulta) está alineada con la información
    reucperada del documento metodológico

    Contexto de referencia:
    {contexto}

    Pregunta/Fragmento a analizar:
    {pregunta}

    
    Objetivo: Analiza la información y determina si el fragmento esta alineado con el contexto dado.
    **No inventes ni des sugernecias**

    Respuesta:

    """),

    

    "evaluar_tecnico_codigo": PromptTemplate(
        input_variables=["contexto", "pregunta"],
        template="""
    Eres un experto en revisión de código y documentación técnica. Valida que el código cumple con las especificaciones técnicas.

    Contexto de referencia:
    {contexto}

    Pregunta/Fragmento a analizar:
    {pregunta}

    Analiza la información y determina si el fragmento esta alineado con el contexto dado.

    Respuesta:

    """),


    "codigo_texto": PromptTemplate(
        input_variables=["codigo"],
        template="""
    ### Instrucciones para el modelo
    Eres un desarrollador experto de modelos PD en SAS. Tu tarea es:
    1. Analizar el siguiente bloque de código SAS.
    2. Describir brevemente su funcionalidad en **3 viñetas**, **máximo 50 palabras cada una**.
    3. Responde siempre con **lenguaje claro y sencillo**, sin alucinaciones.

    Código SAS:
    {codigo}

    Importante: 
    ** No te compliques ni sobrepienses, responde de manera concisa y rápida **
    """), 


    "evaluar_descriptivo_tecnico": PromptTemplate(
        input_variables=["contexto", "pregunta"],
        template=''' 
        Eres un experto en análisis de documentos técnicos y descriptivos relacionados con el SAS.

        **Objetivo:** Validar si la información proporcionada en el contexto (documento técnico) coincide correctamente con la información descrita en la pregunta (documento descriptivo del SAS).

        **Información a analizar (documento técnico):**
        {contexto}

        **Información a comparar (documento descriptivo del SAS):**
        {pregunta}

        **Salida:** Responde de manera clara y concisa si la información del documento técnico coincide con la descripción proporcionada en el documento descriptivo del SAS. 
        Si hay coincidencias o discrepancias, justifica la respuesta basándote únicamente en el contenido proporcionado. No añadas interpretaciones externas ni explicaciones adicionales fuera del análisis.
        '''
        )
 }

# ! PROMPTS PARA COMPARAR TEXTOS 
prompts = {
    "evaluar_requisito": PromptTemplate(
        input_variables=["contexto", "pregunta"],
        template='''
        Eres un modelo experto en análisis de riesgo de crédito.

        **Instrucción:** Evalúa si el requisito dado por la entidad reguladora ({pregunta}) aparece claramente en el contexto proporcionado por el banco ({contexto}).

        **Salida:** Responde únicamente con:
        - "SÍ" si el requisito está claramente presente en el contexto.
        - "NO" si el requisito no está presente o no se puede confirmar.
        '''
    ),
    "evaluar_alineacion_normativa": PromptTemplate(
        input_variables=["contexto", "pregunta"],
        template='''
        Eres un experto en modelos de riesgo de crédito y análisis de fragmentos de texto sobre modelos de riesgo de crédito.

        **Objetivo:** Evaluar si el contexto proporcionado (sobre modelos de riesgo de crédito) se ajusta correctamente a las normativas y regulaciones
        aplicables (supervisor). Justifica si el contexto proporcionado está alineado con la norma y regulación del texto supervisor.

        **Información a analizar (contexto):**
        {contexto}

        **Texto del supervisor (norma):**
        {pregunta}

        **Salida:** Justifica de manera clara y concisa si el contexto se alinea con la normativa (supervisor). Responde EXCLUSIVAMENTE al contexto y texto del supervisor proporcionado,
        sin añadir interpretaciones fuera del análisis.
        La respuesta debe de ser proporcional al tamaño del contexto y del texto del supervisor, no añadas justificaciones fuera del análisis.
        '''
    ), 
    "evaluar_alineacion_normativa_opt" : PromptTemplate(
        input_variables=["contexto", "pregunta"],
        template='''
    Actúas como experto en riesgo de crédito.

    Analiza si el siguiente CONTEXTO está razonablemente alineado con la NORMATIVA proporcionada.

    CONTEXTO:
    {contexto}

    NORMATIVA:
    {pregunta}

    Indica si hay una alineación razonable y justifica brevemente tu conclusión basándote solo en los textos anteriores.
    '''
    )
}

prompt_normativo = prompts['evaluar_alineacion_normativa_opt']

promt_validar_sas = {
    "validar_coincidencia_informacion": PromptTemplate(
        input_variables=["contexto", "pregunta"],
        template=''' 
        Eres un experto en análisis de documentos técnicos y descriptivos relacionados con el SAS.

        **Objetivo:** Validar si la información proporcionada en el contexto (documento técnico) coincide correctamente con la información descrita en la pregunta (documento descriptivo del SAS).

        **Información a analizar (documento técnico):**
        {contexto}

        **Información a comparar (documento descriptivo del SAS):**
        {pregunta}

        **Salida:** Responde de manera clara y concisa si la información del documento técnico coincide con la descripción proporcionada en el documento descriptivo del SAS. 
        Si hay coincidencias o discrepancias, justifica la respuesta basándote únicamente en el contenido proporcionado. No añadas interpretaciones externas ni explicaciones adicionales fuera del análisis.
        '''
        )
        }

prompt_sas = promt_validar_sas['validar_coincidencia_informacion']


# ! PROMPTS PARA TRANSFORMAR CÓDIGO A TEXTO 
prompts_codigo ={
    "codigo_texto_1": PromptTemplate(
        input_variables=['codigo'],
        template='''
        Tu tarea es analizar y explicar de forma clara y estandarizada el siguiente código SAS.

        🎯 Objetivo:
        Describir brevemente el propósito práctico de cada bloque del código SAS, manteniendo un formato uniforme, limpio y fácil de leer.

        📌 Instrucciones:
        1. Divide el código en bloques lógicos (DATA, PROC SQL, pasos de filtrado, transformaciones...).
        2. Resume cada bloque en 1 o 2 frases como máximo.
        3. No repitas el código ni detalles técnicos. Enfócate en qué hace cada bloque y para qué sirve.
        4. Si un bloque genera una tabla, calcula métricas o realiza filtros, indícalo brevemente.
        5. Mantén una estructura **uniforme y elegante** en el formato de salida.

        📄 Formato requerido:

        Bloque 1️⃣  
        **Tipo:** [DATA / PROC SQL / etc.]  
        **Descripción:** [Qué hace este bloque, en lenguaje claro y práctico.]

        Bloque 2️⃣  
        **Tipo:** [DATA / PROC SQL / etc.]  
        **Descripción:** [Qué hace este bloque...]

        ...

        🧠 Sé claro, directo y profesional. El objetivo es que cualquier persona técnica pueda entender la finalidad del código con solo leer tu resumen.

        ---

        Código SAS:
        {codigo}
        '''
    ),

    "codigo_texto_7" : PromptTemplate(
        input_variables=["codigo"],
        template='''
        Eres un desarrollador experto de modelos de PD en SAS.

        Tu tarea es analizar un código en SAS y generar brevemente su descriptivo funcional, teniendo SIEMPRE en cuenta:  
        
        - Explicar con lenguaje CLARO y SENCILLO pero de manera detallada.
        - Limitarte a un MÁXIMO de 200 palabras tu respuesta.

        Código SAS:
        {codigo}

        Importante:

        **Solo escribe el código necesario para detallar tu respuesta**  
        **Usa exclusivamente el código en SAS para tu análisis y no hagas ningún tipo de alunicinación y opinión personal**
        '''
    ), 

    "codigo_texto_8" : PromptTemplate(
        input_variables=["codigo"],
        template='''
        Eres un desarrollador experto de modelos de PD en SAS.

        Tu tarea es analizar un código en SAS y generar brevemente su descriptivo funcional, teniendo SIEMPRE en cuenta:  
        
        - Sintetizar con lenguaje CLARO y SENCILLO el código proporcionado.
        - Limitarte a un MÁXIMO de 200 palabras tu respuesta.

        Código SAS:
        {codigo}

        Importante:

        **Solo escribe el código necesario para detallar tu respuesta**  
        **Usa exclusivamente el código en SAS para tu análisis y no hagas ningún tipo de alunicinación y opinión personal**
        **Puede haber código incompleto, responde solo con lo que puedas ver de manera concisa**
        '''
    ), 
    "codigo_texto_9": PromptTemplate(
        input_variables=["codigo"],
        template="""
    ### Instrucciones para el modelo
    Eres un desarrollador experto de modelos PD en SAS. Tu tarea es:
    1. Analizar el siguiente bloque de código SAS.
    2. Describir brevemente su funcionalidad en **3 viñetas**, **máximo 50 palabras cada una**.
    3. Responde siempre con **lenguaje claro y sencillo**, sin alucinaciones.

    Código SAS:
    {codigo}

    Importante: 
    ** No te compliques ni sobrepienses, responde de manera concisa y rápida **
    """
    ),

    "sas_test": PromptTemplate(
        input_variables=["codigo"],
        template="""
    ### Instrucciones para el modelo
    Eres un experto en análisis estadístico y programación en SAS para entidades bancarias. Tu tarea es:

    1. Analizar el siguiente bloque de código SAS relacionado con los test estadísticos.
    2. Proporciona una **explicación concisa** en **120 palabras** que resuma lo que hace el código.
    3. Usa **terminología estadística adecuada** y asegura que la explicación sea clara y directa, destacando las pruebas estadísticas, las variables involucradas y los resultados clave (como valores p, hipótesis nula, etc.).
    4. Mantén el lenguaje **sencillo y accesible**, sin sobrecomplicar, pero siempre preciso en la interpretación.

    Código SAS:
    {codigo}

    Importante:
    ** No te compliques ni sobrepienses, responde de manera directa y clara.**
    ** No excedas las 120 palabras**
    """

    ),

    "sas_test_4": PromptTemplate(
        input_variables=["codigo"],
        template="""
    Eres un experto en auditoría de modelos de riesgo (PD, LGD) y en traducción de código SAS a texto en español. 
    
    Tu misión es recibir como entrada un bloque de código SAS y devolver:
    
    1. Una descripción en español, clara y concisa, de qué hace cada paso del programa.
    2. La identificación de los test o métricas estadísticas que se están usando (p. ej. Mann-Whitney, Hosmer–Lemeshow, Jeffrey’s, binomial, normal).
    4. Un breve comentario sobre la interpretación de los resultados producidos por ese código (p. ej. “este paso calcula el AUC teórico para comparar con la serie histórica”).

    Si un bloque de código combina varios tests, sepáralos en subapartados.  
    Si una parte del código no encaja exactamente en el índice, clasifícala en “X.Y Otros procedimientos” y explica por qué.

    Código SAS:
    {codigo}

    IMPORTANTE:
    **Responde de manera clara y concisa **
    **Deduce los modelos estadísticos que se están usando**

    """
    ),

    "sas_test_5_claudia": PromptTemplate(
        input_variables=["codigo"],
        template='''
    Analiza el siguiente código SAS y redacta un descriptivo claro y general de lo que hace.

    No inventes nada que no esté presente en el código.
    No hagas suposiciones.
    Si identificas un modelo estadístico concreto, indícalo con su nombre.
    Si estás seguro del propósito del cálculo, especifícalo claramente.
    Si no estás completamente seguro de algún punto, indica expresamente que no se puede saber con certeza.
    No incluyas ejemplos ni nombres concretos (ni de variables ni de segmentos).
    Resume de forma neutral: no asumas qué datos son, ni cómo están estructurados.
    Si detectas segmentaciones o agrupaciones, puedes indicar si parecen buscar homogeneidad, pero sin afirmarlo con certeza.

    Código SAS:
    {codigo}

    Limita la longitud del descriptivo a un máximo de 100 palabras para mantenerlo claro y manejable.
    '''
    ), 

    "sas_test_6": PromptTemplate(
        input_variables=["codigo"],
        template='''
    Analiza el siguiente código SAS con el objetivo principal de identificar e inferir las fórmulas estadísticas subyacentes. Sigue estas instrucciones:

    1. Extrae todas las operaciones matemáticas y funciones estadísticas (por ejemplo, sumas de cuadrados, medias móviles, regresiones, cálculos de varianza, test de hipótesis, etc.) y, cuando sea posible, nómbralas con la terminología estándar (p. ej., “ANOVA de un factor”, “regresión lineal múltiple”, “cálculo de desviación estándar”, “modelo de supervivencia de Cox”, etc.).
    2. No inventes nada que no esté en el código. Si hay fórmulas implícitas (p. ej., cálculo de residuales o transformación logarítmica), ponlas en notación matemática aproximada, como “r_i = y_i – ŷ_i” o “log(X+1)”.
    3. Si reconoces un modelo estadístico concreto, identifícalo por su nombre y, opcionalmente, menciona su función principal (p. ej., “estimación de parámetros”, “análisis de varianza”).
    4. Sé riguroso: si no puedes asegurar con certeza la naturaleza exacta de un cálculo o función, indícalo claramente (“no determinable con certeza”).
    5. Resume de forma neutral y concisa (máximo 100 palabras), sin ejemplos ni nombres de variables, y sin asumir estructura o contenido de los datos.
    
    Código SAS:
    {codigo}

    **Deduce que modelos estadísticos se están utilizando**
    **Limítate a usar un párrafo**
    '''
    ),

    "sas_test_7": PromptTemplate(
        input_variables=["codigo"],
        template='''
    Analiza el siguiente código SAS con estos objetivos:
    1. Identifica todas las operaciones estadísticas y fórmulas implícitas (p. ej., sumas de cuadrados, varianzas, residuales, transformaciones logarítmicas).
    2. Reconoce y nombra los modelos estadísticos específicos (p. ej., “ANOVA de un factor”, “regresión lineal múltiple”, “modelo de Cox”), citando su función principal.
    3. Para cada modelo identificado, añade una breve justificación basada en el contexto del código (por ejemplo, “se usa ANOVA porque compara medias entre k grupos”, “regresión lineal múltiple para cuantificar el efecto conjunto de varias variables continuas”, “Cox para analizar tiempos de falla censurados”).
    4. Sé preciso: no asumas más de lo que muestra el código y marca “no determinable con certeza” donde aplique.
    5. Resume en un solo párrafo de máximo 100 palabras.

    Código SAS:
    {codigo}
    '''
    ), 

    "sas_test_8": PromptTemplate(
        input_variables=["codigo"],
        template='''
    Analiza el siguiente código SAS con estos objetivos:
    1. Identifica todas las operaciones estadísticas y fórmulas implícitas (p. ej., sumas de cuadrados, varianzas, residuales, transformaciones logarítmicas).
    2. Reconoce y nombra los modelos estadísticos específicos (p. ej., “ANOVA de un factor”, “regresión lineal múltiple”, “modelo de Cox”), citando su función principal.
    3. Para cada modelo identificado, añade una breve justificación basada en el contexto del código (por ejemplo, “se usa ANOVA porque compara medias entre k grupos”, “regresión lineal múltiple para cuantificar el efecto conjunto de varias variables continuas”, “Cox para analizar tiempos de falla censurados”).
    4. Sé preciso: no asumas más de lo que muestra el código y marca “no determinable con certeza” donde aplique.
    5. Resume en un solo párrafo de máximo 100 palabras.

    Código SAS:
    {codigo}
    '''
    ),
    
    "sas_test_9": PromptTemplate(
        input_variables=["codigo"],
        template='''
    Tu objetivo es desarrollar en un solo párrafo de no más de 100 palabras, describe de forma fluida y coherente las siguientes observaciones sin usar viñetas: 
        - identifica las operaciones estadísticas y fórmulas implícitas (por ejemplo, sumas de cuadrados, varianzas, residuales, transformaciones logarítmicas), 
        - nombra los modelos estadísticos concretos presentes (por ejemplo, ANOVA de un factor, regresión lineal múltiple, modelo de Cox) indicando su función principal, 
        - aporta una breve justificación contextual para cada modelo (por ejemplo, comparar medias entre grupos, cuantificar efectos de variables continuas o analizar tiempos de falla censurados). 
    Señala con “no determinable con certeza” cualquier aspecto que el código no permita confirmar.

    Código SAS:
    {codigo}
    
    '''
    ),
    
    "sas_test_10": PromptTemplate(
        input_variables=["codigo"],
        template='''
    Tu objetivo es desarrollar en un solo párrafo de no más de 100 palabras, describe de forma fluida y coherente las siguientes observaciones sin usar viñetas: 
        - identifica las operaciones estadísticas y fórmulas implícitas (por ejemplo, sumas de cuadrados, varianzas, residuales, transformaciones logarítmicas), 
        - nombra los modelos estadísticos concretos presentes (por ejemplo, ANOVA de un factor, regresión lineal múltiple, modelo de Cox) indicando su función principal, 
        - aporta una breve justificación contextual para cada modelo (por ejemplo, comparar medias entre grupos, cuantificar efectos de variables continuas o analizar tiempos de falla censurados). 
    Señala con “no determinable con certeza” cualquier aspecto que el código no permita confirmar.

    Código SAS:
    {codigo}

    **Haz solo lo que se te pide, no realices análisis ni recomendaciones aparte**
    **No excedas las 150 palabras**
    '''
    )

}

prompt_codigo_texto = prompts_codigo['codigo_texto_9']


promts_summarizing = {
    "resumen_texto_120": PromptTemplate(
        input_variables=["texto"],
        template="""
    ### Instrucciones para el modelo
    Tu tarea es:
    1. Leer el siguiente texto plano en español.
    2. Sintetizar el objetivo principal del contenido en **120 palabras**, de forma clara y concisa.

    Texto:
    {texto}

    Importante:
    **No te extiendas más allá de 120 palabras, responde de forma precisa y directa.**
    """
        ),

    "resumen_texto_120": PromptTemplate(
        input_variables=["texto"],
        template="""
    ### Instrucciones para el modelo
    Tu tarea es:
    1. Leer el siguiente texto plano en español.
    2. Sintetizar **solo el objetivo principal** del contenido en **120 palabras exactas**, de manera clara y concisa.

    Texto:
    {texto}

    Importante:
    **Responder **solo con un resumen** de **120 palabras exactas****
    """
        )

}

promt_resumen = promts_summarizing['resumen_texto_120']

