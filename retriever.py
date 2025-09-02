from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

class Retriever:
    def __init__(self):
        self.huggingface_embeddings = None
        pass

    def loadDocuments(self, data_dir):
        '''
        Load the PDFs using Langchain's PDF Directory Loader (refer to imported package)
        
        Args:
            data_dir: String path of folder location of PDFs to load

        Returns:
            documents: list of langchain documents
        '''

        loader = PyPDFDirectoryLoader(data_dir)     # provides methods to load and parse multiple PDF documents in a directory
        documents = loader.load()
        return documents

    def splitDocuments(self, documents, chunk_size=700, chunk_overlap=50):
        '''
        Split the loaded documents into smaller chunks using RecursiveCharacterTextSplitter

        https://python.langchain.com/api_reference/text_splitters/base/langchain_text_splitters.base.TextSplitter.html 
        
        Args:
            documents: list of langchain documents
            chunk_size: int, number of characters in a chunk
            chunk_overlap: int, number of characters overlapping between adjacent chunks
        
        Returns:
            document_chunks: list of langchain document chunks
        '''
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
            )
        document_chunks = text_splitter.split_documents(documents)
        return document_chunks

    def createRetriever(self, document_chunks, num_chunks_to_return=4):
        '''
        Initialize the embedding and retriever model using the FAISS vectorstore and huggingface embeddings
        using HuggingFace's BAAI/bge-small-en-v1.5 for the embeddings.

        Args:
            document_chunks: list of langchain document chunks
            num_chunks_to_return: int, number of chunks to retrieve per query

        Returns:
            retriever: langchain VectorStoreRetriever
        '''
        # Sentence transformer model developed by the Beijing Academy of Artificial Intelligence (BAAI)
        self.huggingface_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        vectorstore = FAISS.from_documents(document_chunks, self.huggingface_embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": num_chunks_to_return})
        return retriever
