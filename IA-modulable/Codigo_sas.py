from langchain_community.document_loaders import DirectoryLoader 

import os

from langchain_text_splitters import (
    Language,   # Esta clase proporciona delimitadores específcos para lenguajes pero sas no se encuentra
    RecursiveCharacterTextSplitter,
)

# Tokenizador 
# ==========================================================================================
import sentencepiece as spm
# Tokenizador SentencePiece de Helsinki-NLP/opus-mt-es-en (Apache-2.0), usado para estimar longitudes
SPM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "descargas", "source.spm")
sp = spm.SentencePieceProcessor()
sp.load(SPM_PATH)
# Para diseñar documentos  
# ==========================================================================================
from langchain_core.documents import Document
# Para crear un Word   
# ==========================================================================================
from docx import Document as WordDocument
# Importar la clase que he creado para procesar SAS    
# ==========================================================================================
from X_FUNCTIONS import SASLoader


class ProcesarSAS:
    def __init__(self, sp_model_path=SPM_PATH):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)
        self.historial_respuestas = []

        # 2) Splitter para macros (%MACRO…%MEND;) — sin 'flags', con inline modifiers (?im)
        # Splitter que parte EN BLOQUES DE MACROS completos (%MACRO…%MEND;)
        self.macro_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=[
                "\n%MEND",    # corta justo después de cada %MEND
                "\n%mend"
                "\n%MACRO ",  # corta justo antes de cada %MACRO (incluye el salto)
            ],
            keep_separator="end"  # conserva el separador al final de cada chunk
        )

        # Inicializa el splitter
        # self.sas_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=2000,
        #     chunk_overlap=200,
        #     separators=[
        #     "\nRUN;", "\nQUIT;", "\n%MEND", "\nEND;", "\n%END",
        #     "\nDATA ", "\n%MACRO ", "\n", " "
        #     ],
        #     keep_separator="end"
        # )

    def read_directory(self, path):
        """Carga documentos SAS desde un directorio y añade metadatos."""
        loader = DirectoryLoader(
            path=path,
            glob="**/*.sas",
            show_progress=True,
            loader_cls=SASLoader,  
            loader_kwargs={}
        )
        docs = loader.load()

        for doc in docs:
            source_path = doc.metadata.get("source", "")
            doc.metadata["filename"] = os.path.basename(source_path)

        print(f"Documentos SAS cargados: {len(docs)}")
        return docs
    
    
    def split_by_macros_2(self, text):
        """Split SAS code into macro blocks more accurately.
        
        Handles both %MEND and %MEND macro_name; formats.
        """
        import re
        
        # Create a regex pattern to identify complete macro blocks
        # This accounts for:
        # - Case insensitivity in %MACRO and %MEND
        # - Optional parameters and options in macro definition
        # - Optional semicolon after %MEND
        # - Optional macro name after %MEND
        # - Capture everything between the start and end of a macro
        
        macro_pattern = re.compile(r'''
            (?im)                       # ignore case + multiline
            ^\s*%macro\s+(\w+)\b        # 1) Línea que arranca con %macro + NOMBRE en grupo 1
            (?:\s*\([^)]*\))?\s*;       #    Opcionales: paréntesis con params, luego ;
            (.*?)                       # 2) Contenido de la macro (lazy)
            ^\s*%mend(?:\s+\1)?\s*;     # 3) Línea con %mend (opcionalmente seguido del MISMO_NOMBRE);
            ''',
            re.DOTALL | re.VERBOSE
        )
        
        # Find all macro blocks
        macro_blocks = []
        last_end = 0    # donde terminó el último bloque encontrado
        
        # ! Casos en los que se encuentra el patrón --> lista de casos 
        for match in macro_pattern.finditer(text):
            # If there's code before this macro, add it as a non-macro block
            if match.start() > last_end:
                non_macro_code = text[last_end:match.start()]
                if non_macro_code.strip():  # ? Only add if there's actual information 
                    macro_blocks.append(Document(
                        page_content=f"Non-macro code block:\n{non_macro_code.strip()}",
                        metadata={"block_type": "non_macro"}
                    ))
            
            # Add the macro block
            macro_blocks.append(Document(
                page_content=f"Macro block:\n{match.group(0)}",
                metadata={"block_type": "macro"}
            ))
            
            last_end = match.end()
        
        # Add any remaining code after the last macro
        if last_end < len(text):
            remaining_code = text[last_end:]
            if remaining_code.strip():
                macro_blocks.append(Document(
                    page_content=f"Non-macro code block:\n{remaining_code.strip()}",
                    metadata={"block_type": "non_macro"}
                )) 
        
        return macro_blocks
    

    def split_documents_2(self, documents):
        """Split documents by macros and preserve metadata."""
        split_documents = []
        
        for doc in documents:
            # Process each document to extract macro blocks
            macro_blocks = self.split_by_macros_2(doc.page_content)
            
            # Add original document metadata to each block
            for idx, block in enumerate(macro_blocks, start=1):
                # Preserve original metadata and add block information
                metadata = doc.metadata.copy()
                metadata.update({
                    "block_number": idx,
                    "total_blocks": len(macro_blocks),
                    "block_type": block.metadata["block_type"]
                })
                
                split_documents.append(Document(
                    page_content=block.page_content,
                    metadata=metadata
                ))
        
        return split_documents

    # def split_documents(self, documents):
    #     """Aplica el splitter a los documentos cargados e indica el número de bloque."""
    #     split_documents = []
    #     for doc in documents:
    #         fragments = self.macro_splitter.split_text(doc.page_content)
    #         for idx, fragment in enumerate(fragments, start=1):
    #             bloque_text = f"Bloque {idx}:\n{fragment}"
    #             split_documents.append(Document(page_content=bloque_text, metadata=doc.metadata))
    #     return split_documents


    def agrupar_bloques_por_longitud(self, bloques, max_length):
        """Agrupa bloques respetando un máximo de longitud en tokens."""
        import math
        from prompt import prompt_codigo_texto

        # Calculamos primero los tokens que ocupa el prompt template
        prompt_tokens = len(self.sp.encode(prompt_codigo_texto.template, out_type=int))
        print(f"Tokens del prompt template: {prompt_tokens}")
        
        # La longitud máxima disponible para bloques de código
        max_length_bloques = max_length - prompt_tokens - 200 
        print(f"Tokens disponibles para bloques de código: {max_length_bloques}")
        
        if max_length_bloques <= 0:
            raise ValueError("El prompt template es demasiado largo para el modelo.")

        grupos = []
        grupo_actual, longitud_actual = [], 0

        for bloque in bloques:
            texto = bloque.page_content
            tokens = self.sp.encode(texto, out_type=int)
            longitud_bloque = len(tokens)

            # ! Nueva regla general: calcular splits
            total_tokens = longitud_bloque + prompt_tokens
            if total_tokens > max_length:
                relacion = total_tokens / max_length
                splits = math.ceil(relacion)
            else:
                splits = 1

            # Dividir tokens en `splits` partes
            tamaño_split = len(tokens) // splits
            textos_divididos = [
                self.sp.decode(tokens[i * tamaño_split : (i + 1) * tamaño_split])
                for i in range(splits - 1)
            ]
            textos_divididos.append(self.sp.decode(tokens[(splits - 1) * tamaño_split:]))

            # Procesar cada parte
            for texto_parte in textos_divididos:
                longitud_parte = len(self.sp.encode(texto_parte, out_type=int))

                if longitud_actual + longitud_parte <= max_length_bloques:
                    grupo_actual.append(type(bloque)(page_content=texto_parte))
                    longitud_actual += longitud_parte
                else:
                    if grupo_actual:
                        texto_concatenado = "\n".join(doc.page_content for doc in grupo_actual)
                        grupos.append(texto_concatenado)
                    grupo_actual = [type(bloque)(page_content=texto_parte)]
                    longitud_actual = longitud_parte

        # Guardar el último grupo
        if grupo_actual:
            texto_concatenado = "\n".join(doc.page_content for doc in grupo_actual)
            grupos.append(texto_concatenado)

        return grupos


    def generar_respuestas(self, grupos_para_solicitud):
        """Realizar las solicitudes al agente para transforma código a texto plano"""

        from X_AGENT import codigo, resumen
        respuestas = []
        
        for i, grupo in enumerate(grupos_para_solicitud):
            print(f"📂 Grupo {i+1}, generando respuesta...")
         
            # Incluir el contexto de respuestas anteriores si no es la primera consulta
            if self.historial_respuestas:

                # Añadir el contexto de las respuestas anteriores
                contexto = "\n".join(self.historial_respuestas)
                grupo_con_contexto = f"Esto es una síntesis de la información previa:{contexto}\n Este es el código que tienes que analizar:{grupo}"
            else:
                grupo_con_contexto = grupo   

            # * Código a texto 
            respuesta = codigo.get_answer(grupo_con_contexto)
            respuestas.append(respuesta)

            # * Síntesis 
            resumen_texto = resumen.get_answer(respuesta)        
            self.historial_respuestas.append(resumen_texto)
        
        return respuestas


    def guardar_respuestas_en_word(self, respuestas, ruta_salida="respuestas.docx"):
        doc = WordDocument()
        doc.add_heading("Respuestas generadas por el agente", level=1)

        for i, respuesta in enumerate(respuestas, 1):
            doc.add_heading(f"Consulta {i}:", level=2)
            doc.add_paragraph(respuesta)
            doc.add_paragraph("======================================================================")  # Espacio en blanco entre respuestas

        doc.save(ruta_salida)
        print(f"✅ Respuestas guardadas en: {ruta_salida}")

        return ruta_salida






# class ProcesarSAS:
#     def __init__(self, sp_model_path):
#         self.sp = spm.SentencePieceProcessor()
#         self.sp.load(sp_model_path)

#         # 2) Splitter para macros (%MACRO…%MEND;) — sin 'flags', con inline modifiers (?im)
#         # Splitter que parte EN BLOQUES DE MACROS completos (%MACRO…%MEND;)
#         self.macro_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=2000,
#             chunk_overlap=200,
#             separators=[
#                 "\n%MEND",    # corta justo después de cada %MEND
#                 "\n%mend"
#                 "\n%MACRO ",  # corta justo antes de cada %MACRO (incluye el salto)
#             ],
#             keep_separator="end"  # conserva el separador al final de cada chunk
#         )

#         # Inicializa el splitter
#         # self.sas_splitter = RecursiveCharacterTextSplitter(
#         #     chunk_size=2000,
#         #     chunk_overlap=200,
#         #     separators=[
#         #     "\nRUN;", "\nQUIT;", "\n%MEND", "\nEND;", "\n%END",
#         #     "\nDATA ", "\n%MACRO ", "\n", " "
#         #     ],
#         #     keep_separator="end"
#         # )

#     def read_directory(self, path):
#         """Carga documentos SAS desde un directorio y añade metadatos."""
#         loader = DirectoryLoader(
#             path=path,
#             glob="**/*.sas",
#             show_progress=True,
#             loader_cls=SASLoader,  
#             loader_kwargs={}
#         )
#         docs = loader.load()

#         for doc in docs:
#             source_path = doc.metadata.get("source", "")
#             doc.metadata["filename"] = os.path.basename(source_path)

#         print(f"Documentos SAS cargados: {len(docs)}")
#         return docs
    
#     def split_by_macros(self, text):
#         """Split SAS code into macro blocks more accurately."""
#         import re
        
#         # Create a regex pattern to identify complete macro blocks
#         # This accounts for:
#         # - Case insensitivity in %MACRO and %MEND
#         # - Optional parameters and options in macro definition
#         # - Optional semicolon after %MEND
#         # - Capture everything between the start and end of a macro
        
#         macro_pattern = re.compile(r'''
#             (?im)                       # ignore case + multiline
#             ^\s*%macro\s+(\w+)\b        # 1) Línea que arranca con %macro + NOMBRE en grupo 1
#             (?:\s*\([^)]*\))?\s*;       #    Opcionales: paréntesis con params, luego ;
#             (.*?)                       # 2) Contenido de la macro (lazy)
#             ^\s*%mend\s+\1\s*;          # 3) Línea con %mend MISMO_NOMBRE;
#             ''',
#             re.DOTALL | re.VERBOSE
#         )
        
#         # Find all macro blocks
#         macro_blocks = []
#         last_end = 0    # donde terminó el último bloque encontrado
        
#         # ! Casos en los que se encuentra el patrón --> lista de casos 
#         for match in macro_pattern.finditer(text):
#             # If there's code before this macro, add it as a non-macro block
#             if match.start() > last_end:
#                 non_macro_code = text[last_end:match.start()]
#                 if non_macro_code.strip():  # ? Only add if there's actual information 
#                     macro_blocks.append(Document(
#                         page_content=f"Non-macro code block:\n{non_macro_code.strip()}",
#                         metadata={"block_type": "non_macro"}
#                     ))
            
#             # Add the macro block
#             macro_blocks.append(Document(
#                 page_content=f"Macro block:\n{match.group(0)}",
#                 metadata={"block_type": "macro"}
#             ))
            
#             last_end = match.end()
        
#         # Add any remaining code after the last macro
#         if last_end < len(text):
#             remaining_code = text[last_end:]
#             if remaining_code.strip():
#                 macro_blocks.append(Document(
#                     page_content=f"Non-macro code block:\n{remaining_code.strip()}",
#                     metadata={"block_type": "non_macro"}
#                 )) 
        
#         return macro_blocks
    

#     def split_by_macros_2(self, text):
#         """Split SAS code into macro blocks more accurately.
        
#         Handles both %MEND and %MEND macro_name; formats.
#         """
#         import re
        
#         # Create a regex pattern to identify complete macro blocks
#         # This accounts for:
#         # - Case insensitivity in %MACRO and %MEND
#         # - Optional parameters and options in macro definition
#         # - Optional semicolon after %MEND
#         # - Optional macro name after %MEND
#         # - Capture everything between the start and end of a macro
        
#         macro_pattern = re.compile(r'''
#             (?im)                       # ignore case + multiline
#             ^\s*%macro\s+(\w+)\b        # 1) Línea que arranca con %macro + NOMBRE en grupo 1
#             (?:\s*\([^)]*\))?\s*;       #    Opcionales: paréntesis con params, luego ;
#             (.*?)                       # 2) Contenido de la macro (lazy)
#             ^\s*%mend(?:\s+\1)?\s*;     # 3) Línea con %mend (opcionalmente seguido del MISMO_NOMBRE);
#             ''',
#             re.DOTALL | re.VERBOSE
#         )
        
#         # Find all macro blocks
#         macro_blocks = []
#         last_end = 0    # donde terminó el último bloque encontrado
        
#         # ! Casos en los que se encuentra el patrón --> lista de casos 
#         for match in macro_pattern.finditer(text):
#             # If there's code before this macro, add it as a non-macro block
#             if match.start() > last_end:
#                 non_macro_code = text[last_end:match.start()]
#                 if non_macro_code.strip():  # ? Only add if there's actual information 
#                     macro_blocks.append(Document(
#                         page_content=f"Non-macro code block:\n{non_macro_code.strip()}",
#                         metadata={"block_type": "non_macro"}
#                     ))
            
#             # Add the macro block
#             macro_blocks.append(Document(
#                 page_content=f"Macro block:\n{match.group(0)}",
#                 metadata={"block_type": "macro"}
#             ))
            
#             last_end = match.end()
        
#         # Add any remaining code after the last macro
#         if last_end < len(text):
#             remaining_code = text[last_end:]
#             if remaining_code.strip():
#                 macro_blocks.append(Document(
#                     page_content=f"Non-macro code block:\n{remaining_code.strip()}",
#                     metadata={"block_type": "non_macro"}
#                 )) 
        
#         return macro_blocks
    
#     def split_documents(self, documents):
#         """Split documents by macros and preserve metadata."""
#         split_documents = []
        
#         for doc in documents:
#             # Process each document to extract macro blocks
#             macro_blocks = self.split_by_macros(doc.page_content)
            
#             # Add original document metadata to each block
#             for idx, block in enumerate(macro_blocks, start=1):
#                 # Preserve original metadata and add block information
#                 metadata = doc.metadata.copy()
#                 metadata.update({
#                     "block_number": idx,
#                     "total_blocks": len(macro_blocks),
#                     "block_type": block.metadata["block_type"]
#                 })
                
#                 split_documents.append(Document(
#                     page_content=block.page_content,
#                     metadata=metadata
#                 ))
        
#         return split_documents

#     def split_documents_2(self, documents):
#         """Split documents by macros and preserve metadata."""
#         split_documents = []
        
#         for doc in documents:
#             # Process each document to extract macro blocks
#             macro_blocks = self.split_by_macros_2(doc.page_content)
            
#             # Add original document metadata to each block
#             for idx, block in enumerate(macro_blocks, start=1):
#                 # Preserve original metadata and add block information
#                 metadata = doc.metadata.copy()
#                 metadata.update({
#                     "block_number": idx,
#                     "total_blocks": len(macro_blocks),
#                     "block_type": block.metadata["block_type"]
#                 })
                
#                 split_documents.append(Document(
#                     page_content=block.page_content,
#                     metadata=metadata
#                 ))
        
#         return split_documents

#     # def split_documents(self, documents):
#     #     """Aplica el splitter a los documentos cargados e indica el número de bloque."""
#     #     split_documents = []
#     #     for doc in documents:
#     #         fragments = self.macro_splitter.split_text(doc.page_content)
#     #         for idx, fragment in enumerate(fragments, start=1):
#     #             bloque_text = f"Bloque {idx}:\n{fragment}"
#     #             split_documents.append(Document(page_content=bloque_text, metadata=doc.metadata))
#     #     return split_documents

#     def agrupar_bloques_por_longitud(self, bloques, max_length):
#         """Agrupa bloques respetando un máximo de longitud en tokens."""
#         import math
#         from prompt import prompt_codigo_texto

#         # Calculamos primero los tokens que ocupa el prompt template
#         prompt_tokens = len(self.sp.encode(prompt_codigo_texto.template, out_type=int))
#         print(f"Tokens del prompt template: {prompt_tokens}")
        
#         # La longitud máxima disponible para bloques de código
#         max_length_bloques = max_length - prompt_tokens
#         print(f"Tokens disponibles para bloques de código: {max_length_bloques}")
        
#         if max_length_bloques <= 0:
#             raise ValueError("El prompt template es demasiado largo para el modelo.")

#         grupos = []
#         grupo_actual, longitud_actual = [], 0

#         for bloque in bloques:
#             texto = bloque.page_content
#             tokens = self.sp.encode(texto, out_type=int)
#             longitud_bloque = len(tokens)

#             # ! Nueva regla general: calcular splits
#             total_tokens = longitud_bloque + prompt_tokens
#             if total_tokens > max_length:
#                 relacion = total_tokens / max_length
#                 splits = math.ceil(relacion)
#             else:
#                 splits = 1

#             # Dividir tokens en `splits` partes
#             tamaño_split = len(tokens) // splits
#             textos_divididos = [
#                 self.sp.decode(tokens[i * tamaño_split : (i + 1) * tamaño_split])
#                 for i in range(splits - 1)
#             ]
#             textos_divididos.append(self.sp.decode(tokens[(splits - 1) * tamaño_split:]))

#             # Procesar cada parte
#             for texto_parte in textos_divididos:
#                 longitud_parte = len(self.sp.encode(texto_parte, out_type=int))

#                 if longitud_actual + longitud_parte <= max_length_bloques:
#                     grupo_actual.append(type(bloque)(page_content=texto_parte))
#                     longitud_actual += longitud_parte
#                 else:
#                     if grupo_actual:
#                         texto_concatenado = "\n".join(doc.page_content for doc in grupo_actual)
#                         grupos.append(texto_concatenado)
#                     grupo_actual = [type(bloque)(page_content=texto_parte)]
#                     longitud_actual = longitud_parte

#         # Guardar el último grupo
#         if grupo_actual:
#             texto_concatenado = "\n".join(doc.page_content for doc in grupo_actual)
#             grupos.append(texto_concatenado)

#         return grupos

#     def generar_respuestas(self, grupos_para_solicitud):
#         """Realizar las solicitudes al agente para transforma código a texto plano"""

#         from X_AGENT import codigo
#         respuestas = []
        
#         for i, grupo in enumerate(grupos_para_solicitud):
#             print(f"📂 Grupo {i+1}, generando respuesta...")
            
#             # Consulta con el Agente
#             respuesta = codigo.get_answer(grupo)
#             print(respuesta)
#             respuestas.append(respuesta)
        
#         return respuestas

#     def guardar_respuestas_en_word(self, respuestas, ruta_salida="respuestas.docx"):
#         doc = WordDocument()
#         doc.add_heading("Respuestas generadas por el agente", level=1)

#         for i, respuesta in enumerate(respuestas, 1):
#             doc.add_heading(f"Consulta {i}:", level=2)
#             doc.add_paragraph(respuesta)
#             doc.add_paragraph("======================================================================")  # Espacio en blanco entre respuestas

#         doc.save(ruta_salida)
#         print(f"✅ Respuestas guardadas en: {ruta_salida}")



    ### ALMACENAMIENTO 

    # from docx import Document as DocxDocument

    # # 5. Guardar cada chunk en un archivo Word (.docx) separado
    # output_dir = "chunks_word"
    # os.makedirs(output_dir, exist_ok=True)

    # for idx, doc in enumerate(split_documents):
    #     # Crear nuevo documento Word
    #     word_doc = DocxDocument()
    #     word_doc.add_heading(f'Bloque: {idx}', level=1)
    #     word_doc.add_paragraph("Contenido:")
    #     word_doc.add_paragraph(doc.page_content)

    #     # Guardar el archivo
    #     file_path = os.path.join(output_dir, f"bloque_{idx}.docx")
    #     word_doc.save(file_path)
    #     print(f"Guardado: {file_path}")

    # 3. Crear DataFrame para analizar los splits
    # data = []
    # for idx, doc in enumerate(split_documents):
    #     data.append({
    #         "chunk_id": idx,
    #         "contenido": doc.page_content,  # O simplemente `doc` si no tiene atributo page_content
    #         "metadata": doc.metadata if hasattr(doc, "metadata") else None  # Si tus documentos tienen metadata
    #     })
    # import pandas as pd
    # df = pd.DataFrame(data)

    # # 4. Guardarlo a Excel
    # output_path = "chunks_code3.xlsx"
    # df.to_excel(output_path, index=False)
    # print(f"Se ha guardado el Excel con los splits en: {output_path}")