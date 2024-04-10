# app/rag_assistant.py

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from fastapi import HTTPException
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


class RAGAssistant:
    def __init__(self):
        self.load_env_variables()
        self.setup_prompt_template()
        self.retriever = None  # Define retriever as an instance variable
        # Specify the absolute path of the file
        absolute_path = r'D:\DanyAIApp\code\data\dummy.txt'
        # Get the current working directory
        current_directory = os.getcwd()
        # Calculate the relative path
        relative_path = os.path.relpath(absolute_path, current_directory)
        # default_documents_directory = r"D:\RAG-based-Personal-AI-Assistant\app\data\dummy.txt"
        self.initialize_retriever(relative_path)
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.9)

    def load_env_variables(self):
        """Loads environment variables from .env file."""
        load_dotenv('var.env')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    def setup_prompt_template(self):
        """Sets up the prompt template for chat completions."""
        self.template = """
INSTRUCTIONS:

You are a helpful assistant who will respond to user questions or prompts in a professional manner. You will answer the user's questions or prompts according to their language; if the user's question is in English, then your response will also be in English. However, if the user's question is in French, then your answer will be in French. And follow the same patteren for the rest of the languages.

REMEMBER:
    1 - Do not say like "Hello, Dany! How can I assist you today?"
    Response generally like "Hello there How can I assist you today?", cause you are a general Assistant that will assisting different user simantaneusly.
    2 - Provide response in detail as much as you can. Minimum length of response should be of 500 words for meaningful promts rather than greeting, "Hi,Hey,Hello" etc.
    For Example: Like these kind of important user promts provide detailed response with minimum length of 500 words and maximum 800 words.
        User prompt: what is the diff between nlp and computer vision?
        User prompt: What is AI?
        User prompt: Je débute en marketing digital et je veux optimiser mon e-commerce avec la pub Facebook. Pouvez-vous m'expliquer les KPIs comme le CPA et le ROS et leur calcul ? J'ai besoin d'aide pour utiliser un tableur avec un prix de vente de 17€, et des coûts divers (produit à 3,90€, emballage à 0,80€, livraison à 3,80€, frais de livraison à 2,90€).


EXAMPLES:

    prompt: what is ai?
    Asisstant: AI, or Artificial Intelligence, is like teaching a computer or machine to think and learn like a human. It allows machines to perform tasks that usually require human intelligence, such as recognizing speech, making decisions, and solving problems. Just like humans learn from experience, AI improves over time by learning from data.

    prompt: Qu'est-ce que l'IA ?
    Asisstant: L'IA, ou Intelligence Artificielle, c'est comme apprendre à un ordinateur ou à une machine à penser et à apprendre comme un humain. Elle permet aux machines d'effectuer des tâches qui nécessitent généralement de l'intelligence humaine, telles que reconnaître la parole, prendre des décisions et résoudre des problèmes. Tout comme les humains apprennent de l'expérience, l'IA s'améliore au fil du temps en apprenant à partir des données.

FORMATTING:
    1 - Use Bullet Points

<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""
        self.prompt_template = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=self.template,
        )

    def initialize_retriever(self, directory_path):
        """Initializes the retriever with documents from the specified directory path."""
        loader = TextLoader(directory_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()

        Pinecone(api_key=self.pinecone_api_key, environment='gcp-starter')
        vectbd = PineconeVectorStore.from_documents(
            docs, embeddings, index_name=self.pinecone_index_name)
        self.retriever = vectbd.as_retriever()

    def generate_response(self, query):
        if not self.retriever:
            # Replace HTTPException with a normal exception
            raise Exception("Retriever not initialized")

        # Create an instance of the RetrievalQA chain (this should be synchronous)
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.retriever,
            chain_type_kwargs={"verbose": False, "prompt": self.prompt_template,
                               "memory": ConversationBufferMemory(memory_key="history", input_key="question")}
        )

        # Use the chain to invoke the model and generate a response (this should be synchronous)
        assistant_response = chain.invoke(query)
        return assistant_response.get('result', 'No response generated')

    def finetune(self, file_path):
        """Determines the document type and uses the appropriate loader to fine-tune the model."""
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        elif file_path.endswith('.csv'):
            loader = CSVLoader(file_path=file_path)
        elif file_path.endswith('.xlsx'):
            loader = UnstructuredExcelLoader(file_path, mode="elements")
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Unsupported file type.")

        documents = loader.load_and_split() if hasattr(
            loader, 'load_and_split') else loader.load()

        self.process_documents(documents)

    def process_documents(self, documents):
        """Process and index the documents."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        Pinecone(api_key=self.pinecone_api_key, environment='gcp-starter')
        vectbd = PineconeVectorStore.from_documents(
            docs, embeddings, index_name=self.pinecone_index_name)
        self.retriever = vectbd.as_retriever()
