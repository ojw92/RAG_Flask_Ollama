import os
from dotenv import load_dotenv
from typing import Any
from langchain_community.llms import Ollama
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.language_models import BaseLLM
from langchain_core.runnables import Runnable
from gradio_client import Client
from pydantic import PrivateAttr
from langchain_core.outputs import Generation, LLMResult
from retriever import Retriever
from pipeline import get_llm_port

class GradioLLMWrapper(BaseLLM, Runnable):
    _client: Any = PrivateAttr()

    def __init__(self, space_name: str, hf_token: str):
        super().__init__()
        object.__setattr__(self, "_client", Client(space_name, hf_token=hf_token))

    def _call(self, prompt: str, **kwargs: Any) -> str:
        result = object.__getattribute__(self, "_client").predict(prompt, api_name="/predict")
        return result

    def invoke(self, input: str, **kwargs: Any) -> str:
        return self._call(input, **kwargs)

    def _generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        return [self._call(prompt, **kwargs) for prompt in prompts]

    def generate(self, prompts: list[str], **kwargs: Any) -> LLMResult:
        generations = self._generate(prompts, **kwargs)
        return LLMResult(
            generations=[[Generation(text=gen)] for gen in generations]
        )

    @property
    def _llm_type(self) -> str:
        return "gradio-flan"

class RAG_Chain:
    def __init__(self, data_dir, llm_type="gradio_flan", init_retriever=True, llm_model="llama3.2", llm_ag=None): 
        '''
        Initializes the RAG chain by selecting an LLM backend and loading the document retriever system.
        Load the api-key & space name from the .env file and initialize self.llm and self.retriever_system
        Get an API Access Token to access the HuggingFace models.

        Args:
            data_dir: String path of folder location of PDFs to load
            llm_ag: Custom LLM agent for injecting a pre-initialized model manually.
            llm_type: String to determine which llm to use (for Q8)
            init_retriever: boolean switch on whether to instantiate retriever (for Q8)
            llm_model: String specifiying which ollama model to use (for Q8)

        Initialize:
            self.llm: LLM from HuggingFaceHub. 
            self.retriever_system: retrieval system implemented in retriever.py

        Returns:
            None
        '''

        if llm_ag is not None:
            self.llm = llm_ag

        elif llm_type == "ollama_only":
            self.set_ollama_only(llm_model=llm_model)

        elif llm_type == "flask_ollama":
            self.set_flask_ollama(llm_model=llm_model)

        elif llm_type == "gradio_flan":
            load_dotenv()
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            space_name = os.getenv("GRADIO_SPACE_NAME")
            self.llm = GradioLLMWrapper(
                space_name=space_name,
                hf_token=api_key
            )


        # Load the retriever system
        if init_retriever:
            self.retriever_system = self.init_retriever_system(data_dir)

    def set_ollama_only(self, llm_model="llama3.2"):
        '''
        Initialize self.llm using Ollama and set model to the llm_model

        Args:
            llm_model: String specifiying which ollama model to use

        Initialize:
            self.llm: llm_model from ollama

        Returns:
            None
        Note: See the Jupyter Notebook for documentation on the imported Ollama wrapper
        '''
        self.llm = Ollama(model=llm_model)
        
    def set_flask_ollama(self, llm_model="llama3.2", api_key="None"):
        '''
        Initialize self.llm using the OpenAI wrapper and set model to the llm_model. 
        The required fields are openai_api_base, openai_api_key, and model_name.
        Note that although the implementation does not require a key, this is still a required field and needs a placeholder.

        Args:
            llm_model: String specifiying which ollama model to use

        Initialize:
            self.llm: llm_model via Flask Ollama

        Returns:
            None
        Note: get_llm_port() from the pipeline.py may be helpful
        '''
        port = get_llm_port()
        self.llm = OpenAI(openai_api_base = f"http://localhost:{port}/",
                          openai_api_key = api_key,
                          model_name = llm_model
                          )

    def query_the_llm(self, question):
        """
        Invokes a question to the RAG's LLM without any supporting documents.
        """
        response = self.llm.invoke(question)
        return response

    def init_retriever_system(self, data_dir):
        '''
        Initialize the retriever system by instantiating the retriever implementation from retriever.py and loading the documents in the PDF directory.
        Split the loaded documents into chunks and use the chunks to create and return the VectorStoreRetriever

        Args:
            data_dir: String path of folder location of PDFs to load

        Returns:
            retriever: langchain VectorStoreRetriever

        Note: Refer to the local test code in 1.2) to see an example of how a retriever is initialized.
        The chunk_size, chunk_overlap, and num_chunks_to_return can be omitted to use their default values.
        '''
        retriever = Retriever()
        docs = retriever.loadDocuments(data_dir)
        docs_split = retriever.splitDocuments(docs)
        retriever = retriever.createRetriever(docs_split)
        return retriever

    def createPrompt(self, question):
        '''
        Define the prompt template and return a formatted prompt using the template and question argument.

        Args:
            question: Dictionary with the following keys: 'question', 'A', 'B', 'C', 'D'. See notebook for example
        
        Returns:
            formatted_prompt: The question and answer choices reformatted using the prompt template to use to query the LLM.
        '''
        prompt_template = ChatPromptTemplate.from_template(
            "Question: {question}\nA: {A}\nB: {B}\nC: {C}\nD: {D}\nPlease select the best answer and explain your choice."
            )
        formatted_prompt = prompt_template.format(**question)
        return formatted_prompt

    def createRAGChain(self):
        '''
        Build the RAG pipeline using the RetrievalQA chain. Make sure to pass the LLM (self.llm) and retriever system (self.retriever_system).
        Refer to: https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html#langchain.chains.retrieval_qa.base.RetrievalQA.from_chain_type

        Args:
            None
            
        Returns:
            qa_chain: BaseRetrievalQA used to answer multiple choice questions.
        '''
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever_system,
            return_source_documents=True
            )
        return qa_chain
    