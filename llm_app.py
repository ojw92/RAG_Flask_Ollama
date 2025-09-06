from flask import Flask, request, jsonify
import requests
import json
import socket
import ollama

class llmApp():
    def __init__(self):
        """
        Initialize the flask app, port that flask runs on, and setup routes.
        Do not change Ollama API's url.
        """
        self.app = Flask(__name__)
        self.OLLAMA_URL = "http://127.0.0.1:11434/api/generate"     # OLLAMA API, do not change
        self.port = self.find_available_port()
        self.setup_routes()

    def setup_routes(self):
        """
        Define the setup routes for flask app.
        Routes:
            - GET `/`: Returns a message that app is running.
            - POST `/completions`: Accepts a prompt and parameters, and then queries for a response.
        """

        @self.app.route("/", methods=["GET"])
        def home():
            """
            Homepage for the LLM API. 
            NOTE : This function is given and does not have to be implemented.
            """
            return jsonify({"message": "LLM API is running."})

        @self.app.route("/completions", methods=["POST"])
        def generate():
            """
            Handle requests to generate text using Ollama's LLM. 

            Accepts a JSON payload that *may* contain:
                model (str): Ollama model to be used (default to llama3.2)
                prompt (str): User prompt (default to empty string). May assume that the prompt will be either a sentence of a list containing exactly one sentence.
                temperature (float): Control the randomness of the response. Make sure to set up the temperature to 0.0.
                top_p (float): A threshold probability to select the the top tokens whose cumulative probability exceeds the threshold (default to 0.8). 
                max_tokens (int): Maximum tokens that model generates (default to 50)
                do_sample: boolean (default to false)
            
            Return:
                JSON response: containing the model used and the full response following the OpenAI response structure
            """

            data = request.json

            # 1: Extract relevant information from data, .get may be useful.
            model_used = data.get("model", "llama3.2")
            prompt = data.get("prompt", "")
            if isinstance(prompt, list):            # APIs receive the prompt as string or list of string
                prompt = prompt[0] if len(prompt) > 0 else ""
            temperature = data.get("temperature", 0.0)
            top_p = data.get("top_p", 0.8)
            max_tokens = data.get("max_tokens", 50)
            do_sample = data.get("do_sample", False)


            # 2: Recombine the extracted data into a json format accepted by Ollama. 
            # https://github.com/ollama/ollama/blob/main/docs/api.md#request-json-mode may be helpful
            json_input = {
                "model": model_used,
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": True      # stream output as Ollama expects
                }
            


            response = requests.post(
                self.OLLAMA_URL,
                json=json_input,
            )


            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_obj = json.loads(line)
                        full_response += json_obj.get("response", "")
                    except requests.exceptions.JSONDecodeError:
                        continue  # Ignore malformed JSON lines


            # 3: Create the return output, include the model used and full response in the following format json format. Note that usage is required as a placeholder only.
            # "model" <- model_used, 
            # "choices"[0]["text"] <- full_response
            # "usage"["prompt_tokens"] <- 0
            # "usage"["completion_tokens"] <- 0
            # "usage"["total_tokens"] <- 0
            # Hint: jsonify  may be helpful
            output = jsonify({
                    "model": model_used,
                    "choices": [{"text": full_response}],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                        }
                    })



            return output

    def find_available_port(self):
        """
        Find available port to host local LLM.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))  
            return s.getsockname()[1]  
            
    def run(self):
        """
        Run the flask app on the available port and write it on llm_port.txt.
        """
        print(f"LLM is running on port {self.port}")
        with open("llm_port.txt", "w") as f:
            f.write(str(self.port))
        self.app.run(debug=True, port = self.port, use_reloader=False)

if __name__ == "__main__":
    llm_app = llmApp()
    llm_app.run()
