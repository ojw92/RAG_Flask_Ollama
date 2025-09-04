## This file will be used to create the ouptuts in the Jupyter Notebook

import hashlib
import subprocess
import json

# Predefined parameters
temperature = 0.1
top_p = 0.1
max_tokens = 50

def get_llm_port():
    """
    Read the LLM port from 'llm_port.txt' and returns the port number, or None if the file is missing.
    """
    try:
        with open("llm_port.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("Error: 'llm_port.txt' not found. Ensure the LLM is running.")
        return None

def get_rag_port():
    """
    Read the RAG port from 'rag_port.txt' and returns the port number, or None if the file is missing.
    """
    try:
        with open("rag_port.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("Error: 'rag_port.txt' not found. Ensure the RAG app is running.")
        return None

# Function to hash the answer using MD5
def hash_answer(answer, sid=""):
    answer = sid + json.dumps(answer[0])
    return hashlib.md5(answer.encode()).hexdigest()

def query_ollama(question, model="llama3.2"):
    # Prepare the curl command as a string

    data = {
    "model": model,
    "prompt": question,
    "stream": False,
    "options": {
        "temperature": 0
        }
    }

    curl_command = [
        "curl",
        "-X", "POST",
        "http://127.0.0.1:11434/api/generate",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(data)
    ]

    # Execute the curl command and capture the output
    result = subprocess.run(curl_command, capture_output=True, text=True)

    if result.returncode == 0:
        # Return the response from the server (JSON formatted)
        return json.loads(result.stdout).get("response")
    else:
        return f"Error: {result.returncode}, {result.stderr}"

def query_flask_ollama(question, model="llama3.2"):
    # Prepare the curl command as a string
    llm_port = get_llm_port()
    llm_api_url = f"http://127.0.0.1:{llm_port}/completions"

    data = {
        "model": model,
        "prompt": question,
        "stream": False,
        "temperature": 0
    }
    
    curl_command = [
        "curl",
        "-X", "POST",
        llm_api_url,
        "-H", "Content-Type: application/json",
        "-d", json.dumps(data)
    ]

    # Execute the curl command and capture the output
    result = subprocess.run(curl_command, capture_output=True, text=True)

    if result.returncode == 0:
        # Return the response from the server (JSON formatted)
        return json.loads(result.stdout).get("choices")[0]["text"]
    else:
        return f"Error: {result.returncode}, {result.stderr}"

def query_flask_rag(question, model="llama3.2", max_tokens=100, temperature=0):
    # Prepare the curl command as a string
    rag_port = get_rag_port()
    rag_api_url = f"http://127.0.0.1:{rag_port}/query"

    data = {
        "model": model,
        "query": question,
        "stream": False,
        "temperature": 0
    }
    
    curl_command = [
        "curl",
        "-X", "POST",
        rag_api_url,
        "-H", "Content-Type: application/json",
        "-d", json.dumps(data)
    ]

    # Execute the curl command and capture the output
    result = subprocess.run(curl_command, capture_output=True, text=True)

    if result.returncode == 0:
        # Return the response from the server (JSON formatted)
        return json.loads(result.stdout).get("choices")
    else:
        return f"Error: {result.returncode}, {result.stderr}"

def main():

    # List of questions to query
    with open('./data/questions.json', 'r') as file:
        questions = json.load(file)

    hashed_answers = {}
    # Query the server and process the responses
    for idx, (q_key, question) in enumerate(questions.items(), 1):
        print(f"Question {idx}: {question}")
        answer = query_flask_rag(question)[0]["text"]
        
        if answer:
            print(f"Answer {idx}: {answer}")
            # Hash the answer using MD5
            hashed_answer = hash_answer(answer)
            print(f"Hashed Answer {idx} (MD5): {hashed_answer}")
            hashed_answers[q_key] = hashed_answer
        else:
            print(f"Error retrieving answer for Question {idx}")
        
        print("-" * 80)

    # Write hashed answers to hashed_answers.json
    with open('hashed_answers.json', 'w') as file:
        json.dump(hashed_answers, file, indent=4)

if __name__ == "__main__":
    main()
