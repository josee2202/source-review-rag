from config import config
from langchain_ollama.llms import OllamaLLM

# config parameters
cfg_param = config.cfg_param

host = cfg_param['ollama']['host']
port = cfg_param['ollama']['port']
ollama_endpoint = f"http://{host}:{port}"

llm_model_name = cfg_param['llm']['text']['name']
llm_max_tokens = cfg_param['llm']['text'].get('max_tokens', 512)  # evita generaciones sin fin (num_predict)
llm_timeout_s = cfg_param['llm']['text'].get('timeout_s', 300)


class Agente:
    """Genera el comentario opcional. El prompt se fija según la modalidad de cada petición."""

    def __init__(self, ollama_endpoint, model_name, temperature=0):
        self.llm = OllamaLLM(base_url=ollama_endpoint, model=model_name, temperature=temperature,
                             num_predict=llm_max_tokens, client_kwargs={"timeout": llm_timeout_s})
        self.prompt = None  # Inicializar sin prompt
        self.chain = None   # Inicializar sin chain

    def set_prompt(self, prompt):
        """
        Establece el prompt y recrea la cadena.

        Args:
            prompt: El template de prompt a usar
        """
        self.prompt = prompt
        # think=False: un modelo con razonamiento (Qwen3.5) gasta si no todo num_predict pensando y el comentario queda vacío.
        # Los modelos sin razonamiento lo aceptan igual (comprobado con llama3.2:1b y Ollama 0.40.14).
        self.chain = self.prompt | self.llm.bind(think=False)

    def get_answer(self, contexto, pregunta):
        """
        Genera un comentario basado en los pasajes recuperados y el fragmento revisado.

        Args:
            contexto (str): Pasajes recuperados de la referencia.
            pregunta (str): Fragmento del texto que se revisa.

        Returns:
            str: Comentario generado por el modelo.
        """
        if self.chain is None:
            raise ValueError("Debes establecer un prompt primero usando set_prompt()")

        input_data = {
            "contexto": contexto,
            "pregunta": pregunta
        }

        self.answer = self.chain.invoke(input_data)
        return self.answer


agente_dinamico = Agente(ollama_endpoint, llm_model_name)
