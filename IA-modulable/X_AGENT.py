from config import config
from langchain_ollama.llms import OllamaLLM
from prompt import prompt_normativo, prompt_codigo_texto, prompt_sas, promt_resumen

# config parameters
cfg_param = config.cfg_param

host = cfg_param['ollama']['host']
port = cfg_param['ollama']['port']
ollama_endpoint = f"http://{host}:{port}"

llama32_1b_model_name = cfg_param['llm']['text']['name']  # Pequeño 
llm_max_tokens = cfg_param['llm']['text'].get('max_tokens', 1024)  # evita generaciones sin fin (num_predict)

# Agente 
# class Agente:
#     def __init__(self, ollama_endpoint, model_name, prompt, temperature = 0):
#         self.llm = OllamaLLM(base_url=ollama_endpoint, model=model_name, temperature = temperature)
#         self.prompt = prompt
#         self.chain = self.prompt | self.llm
        
#     def get_answer(self, contexto, pregunta):
#         """
#         Genera una respuesta del modelo basada en el contexto y la pregunta proporcionados.
        
#         Args:
#             contexto (str): Información contextual recuperada de Qdrant u otra fuente.
#             pregunta (str): La pregunta que se quiere responder.
        
#         Returns:
#             str: Respuesta generada por el modelo.
#         """
#         # Formatear la entrada combinando contexto y pregunta
#         input_data = {
#             "contexto": contexto,
#             "pregunta": pregunta
#         }
        
#         # Invocar la cadena con los datos formateados
#         self.answer = self.chain.invoke(input_data)
#         return self.answer

# # Crear agente
# agente_normativo = Agente(ollama_endpoint, llama32_1b_model_name, prompt_normativo)

class Agente:
    def __init__(self, ollama_endpoint, model_name, temperature=0):
        self.llm = OllamaLLM(base_url=ollama_endpoint, model=model_name, temperature=temperature,
                             num_predict=llm_max_tokens)
        self.prompt = None  # Inicializar sin prompt
        self.chain = None   # Inicializar sin chain
        
    def set_prompt(self, prompt):
        """
        Establece el prompt y recrea la cadena.
        
        Args:
            prompt: El template de prompt a usar
        """
        self.prompt = prompt
        self.chain = self.prompt | self.llm
        
    def get_answer(self, contexto, pregunta):
        """
        Genera una respuesta del modelo basada en el contexto y la pregunta proporcionados.
        
        Args:
            contexto (str): Información contextual recuperada de Qdrant u otra fuente.
            pregunta (str): La pregunta que se quiere responder.
        
        Returns:
            str: Respuesta generada por el modelo.
        """
        if self.chain is None:
            raise ValueError("❌ Debes establecer un prompt primero usando set_prompt()")
            
        # Formatear la entrada combinando contexto y pregunta
        input_data = {
            "contexto": contexto,
            "pregunta": pregunta
        }
        
        # Invocar la cadena con los datos formateados
        self.answer = self.chain.invoke(input_data)
        return self.answer

agente_dinamico = Agente(ollama_endpoint, llama32_1b_model_name)




# Agente 
class Agente_codigo:
    def __init__(self, ollama_endpoint, model_name, temperature = 0):
        self.llm = OllamaLLM(base_url=ollama_endpoint, model=model_name, temperature = temperature,
                             num_predict=llm_max_tokens)
        self.prompt = prompt_codigo_texto
        self.chain = self.prompt | self.llm
        
    def get_answer(self, codigo):
        """
        Genera una descripción del código SAS proporcionado en lenguaje natural.
        
        Args:
            codigo (str): El código SAS que se desea describir.
        
        Returns:
            str: Descripción del código SAS en texto plano.
        """
        # Formatear la entrada con el código SAS
        input_data = {
            "codigo": codigo
        }
         
        # Invocar la cadena con el código proporcionado
        self.answer = self.chain.invoke(input_data)
        return self.answer
    
codigo = Agente_codigo(ollama_endpoint, llama32_1b_model_name)


# Agente 
class Agente_summarizer:
    def __init__(self, ollama_endpoint, model_name, temperature = 0):
        self.llm = OllamaLLM(base_url=ollama_endpoint, model=model_name, temperature = temperature,
                             num_predict=llm_max_tokens)
        self.prompt = promt_resumen
        self.chain = self.prompt | self.llm
        
    def get_answer(self, texto):
        """
        Genera una descripción del código SAS proporcionado en lenguaje natural.
        
        Args:
            codigo (str): El código SAS que se desea describir.
        
        Returns:
            str: Descripción del código SAS en texto plano.
        """
        # Formatear la entrada con el código SAS
        input_data = {
            "texto": texto
        }
         
        # Invocar la cadena con el código proporcionado
        self.answer = self.chain.invoke(input_data)
        return self.answer
    
resumen = Agente_summarizer(ollama_endpoint, llama32_1b_model_name)


