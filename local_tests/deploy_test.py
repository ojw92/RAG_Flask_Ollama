class Deploy_Test():
    def __init__(self):

        # Query ollama with temperature 0, llama3.2 (8.1.1 and 8.2.1)
        self.ollama_query = "True or False, an apple is a fruit"
        self.ollama_response = "TRUE. An apple is indeed a type of fruit that grows on trees in the Rosaceae family."


        self.ollama_rag_query = "True or False, a bird is a bat."
        self.ollama_rag_response = "False. Birds and bats are two distinct groups of animals that belong to different classes (Aves for birds and Chiroptera for bats) and have many physical and behavioral differences."
        

        self.flask_ollama_rag_query = "True or False, a bird is a bat."
        self.flask_ollama_rag_response = "False. Birds and bats are two distinct groups of animals that belong to different classes (Aves for birds and Chiroptera for bats) and have many physical and behavioral differences."


        self.flask_rag_query = "Does Glove use word similarity?"
        self.flask_rag_response = "Yes, according to the text, GloVe uses word-word co-occurrence counts, which implies that it does use word similarity as a basis for generating word representations."
