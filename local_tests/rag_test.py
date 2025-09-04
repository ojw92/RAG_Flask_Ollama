class RAG_Test:
    def __init__(self):
        # Some tests for the retriever


        # Tests for the QA chain (entire RAG pipeline)
        self.question1 = {
            'question': "What is the main contribution of the Transformer architecture?",
            'A': "It introduces convolutional layers for sequence tasks.",
            'B': "It improves word embeddings using context.",
            'C': "It removes recurrence and uses self-attention mechanisms.",
            'D': "It uses RNNs for language modeling."
        }
        self.answer1 = 'C'

        self.question2 = {
            'question': "The Transformer architecture relies heavily on which mechanism to process information?",
            'A': "Convolution",
            'B': "Self-attention",
            'C': "Pooling",
            'D': "Dropout"
        }
        self.answer2 = 'B'

        self.question3 = {
            'question': "What is the primary innovation introduced by BERT?",
            'A': "Unidirectional language models",
            'B': "Bidirectional training of transformers",
            'C': "Convolutional embeddings",
            'D': "Memory networks"
        }
        self.answer3 = 'B'

        self.question4 = {
            'question': "BERT is pre-trained on which type of task?",
            'A': "Named entity recognition",
            'B': "Image classification",
            'C': "Machine translation",
            'D': "Next sentence prediction and masked language modeling"
        }
        self.answer4 = 'D'

        self.question5 = {
            'question': "What is the key innovation introduced in the paper 'Attention is All You Need'?",
            'A': "The Transformer architecture",
            'B': "Recurrent Neural Networks",
            'C': "Long Short-Term Memory (LSTM)",
            'D': "Bahdanau attention"
        }
        self.answer5 = 'A'

        self.question6 = {
            'question': "GloVe primarily uses which mathematical construct to represent word vectors?",
            'A': "Recurrent networks",
            'B': "Singular value decomposition",
            'C': "Word co-occurrence matrix",
            'D': "Self-attention mechanism"
        }
        self.answer6 = 'C'

        self.question7 = {
            'question': "What is the key contribution of the paper 'Improving Language Understanding by Generative Pre-Training (GPT)'?",
            'A': "Introduction of attention mechanisms",
            'B': "Building deep convolutional networks",
            'C': "Combining LSTM and CNN architectures",
            'D': "Pre-training on large datasets followed by fine-tuning for specific tasks"

        }
        self.answer7 = 'D'

        self.question8 = {
            'question': "What is the key concept introduced in the RAG framework?",
            'A': 'Combining retrieval-based methods with generative models',
            'B': 'Unsupervised pre-training on large corpora',
            'C': 'End-to-end training of neural networks',
            'D': 'Using reinforcement learning for text generation'
        }
        self.answer8 = 'A'

        self.local_test_questions = [
            self.question1,
            self.question2, 
            self.question3, 
            self.question4, 
            self.question5, 
            self.question6, 
            self.question7, 
            self.question8
        ]

        self.local_test_answers = [
            self.answer1,
            self.answer2,
            self.answer3,
            self.answer4,
            self.answer5,
            self.answer6,
            self.answer7,
            self.answer8
        ]

