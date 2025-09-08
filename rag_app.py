from flask import Flask, request, jsonify
from rag_chain import RAG_Chain
import requests
import socket

class ragApp():
    def __init__(self):
        """
        Initialize the ragApp class and set up the local LLM and RAG components.
        Attributes:
            self.app (Flask): Flask application instance.
            self.data_dir (str): PDF folder directory for retrieval.
            self.llm_url (str): The API endpoint for sending queries to the local LLM.
            self.rag (RAG_Chain): Instance of the RAG chain for document retrieval.
            self.port (int): The available port number for the Flask app.
        """

        self.app = Flask(__name__)

        # Initialize RAG and QA Chain
        self.data_dir = './data/papers/'
        self.rag = RAG_Chain(self.data_dir, llm_type="flask_ollama")
        self.qa_chain = self.rag.createRAGChain()

        # Port and Route Setup
        self.port = self.find_available_port()
        self.setup_routes()

    def setup_routes(self):
        """
        Define the Flask API routes for handling user queries.
        Routes:
            - `/query`: Handles document retrieval and LLM queries.
        """


        @self.app.route("/query", methods=["POST"])
        def query_rag():
            """
            Handles user queries by querying the RAG and retrieving relevant documents and LLM response.

            Accepts a JSON payload that *may* contain:
                model (str): Ollama model to be used (default to llama3.2)
                user_query (str): The question to ask the RAG (default to empty string). 
                temperature (float): Control the randomness of the response (default to 0)
                top_p (float): A threshold probability to select the the top tokens whose cumulative probability exceeds the threshold (default to 0.8)
                max_tokens (int): Maximum tokens the model generates (default to 100)
                do_sample: boolean (default to false)
            
            Return:
                JSON response: containing the model used and the full response following the OpenAI response structure
            """

            data = request.json

            # 1: Extract relevant information from data, .get may be useful
            user_query = data.get("user_query", "")
            model_used = data.get("model", "llama3.2")
            temperature = data.get("temperature", 0.0)
            top_p = data.get("top_p", 0.8)
            max_tokens = data.get("max_tokens", 100)
            do_sample = data.get("do_sample", False)


            # 2: Set the RAG LLM parameters to the extracted data settings
            self.rag.set_flask_ollama(model_used)
            self.rag.llm.temperature = temperature
            self.rag.llm.top_p = top_p
            self.rag.llm.max_tokens = max_tokens


            # 3: Query the RAG QA Chain
            response_data = self.qa_chain({"query": user_query})
            


            # Convert retrieved documents into a JSON-serializable format
            retrieved_docs = response_data.get("source_documents", [])
            retrieved_docs_serializable = [
                {"page_content": doc.page_content, "metadata": doc.metadata} for doc in retrieved_docs
            ]


            # 4: Create the return output. Return the following into the json format. Note that usage is included as a placeholder only
            # "choices"[0]["model"] <- model_used, 
            # "choices"[0]["query"] <- user_query
            # "choices"[0]["text"] <- result from response_data (default to "No Response")
            # "choices"[0]["retrieved_context"] <- retrieved_docs_serializable
            # "usage"["prompt_tokens"] <- 0
            # "usage"["completion_tokens"] <- 0
            # "usage"["total_tokens"] <- 0
            # Hint: jsonify may be helpful
            output = jsonify({
                "choices": [{
                    "model": model_used,
                    "query": user_query,
                    "text": response_data.get("result", "No Response"),
                    "retrieved_context": retrieved_docs_serializable
                    }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                    }
                })
            


            return output


    def find_available_port(self):
        """
        Find an available port to run the RAG Flask app.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def run(self):
        """
        Run the Flask app and save the port number to 'rag_port.txt'.
        """
        print(f"RAG App is running on port {self.port}")
        with open("rag_port.txt", "w") as f:
            f.write(str(self.port))
        self.app.run(port=self.port, debug=True, use_reloader=False)


if __name__ == "__main__":
    rag_app = ragApp()
    rag_app.run()
