from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
import pinecone
import os

class PineconeInitializer:

    def __init__(self):

        self.index_name = "lawbot-index"
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.pinecone = pinecone.Pinecone(api_key=os.environ['PINECONE_API_KEY'])


        # self.deleteIndex()
        self.upsert()
        
    def upsert(self):

        loader = DirectoryLoader("Database/")
        documents = loader.load()

        docs = self.text_splitter.split_documents(documents)

        self.pinecone.create_index(self.index_name,dimension=1536,spec=pinecone.ServerlessSpec(cloud = "aws",region="us-east-1"))

        PineconeVectorStore.from_documents(
            docs,
            index_name=self.index_name,
            embedding=self.embeddings
        )

    def deleteIndex(self):

        self.pinecone.delete_index(self.index_name)

PineconeInitializer()
