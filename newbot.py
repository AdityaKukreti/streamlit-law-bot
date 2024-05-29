import os
import pinecone
import time
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI  
from langchain.chains import RetrievalQA,RetrievalQAWithSourcesChain


class LawBot:

    def __init__(self):
        self.pc = pinecone.Pinecone(api_key=os.environ['PINECONE_API_KEY'])
        self.spec = pinecone.ServerlessSpec(cloud = "aws", region = 'us-east-1')

        self.index_name = 'lawbot-index'

        
        self.index = self.pc.Index(self.index_name)

        self.model_name = "text-embedding-ada-002"
        self.embeddings = OpenAIEmbeddings(model=self.model_name,openai_api_key = os.environ['OPENAI_API_KEY'])

        self.text_field = "text"  
        self.vectorstore = PineconeVectorStore(  
            self.index, self.embeddings, self.text_field  
        )  



        self.llm = ChatOpenAI(  
            openai_api_key=os.environ['OPENAI_API_KEY'],  
            model_name='gpt-3.5-turbo',  
            temperature=0.0  
        )  
        self.qa = RetrievalQAWithSourcesChain.from_chain_type(  
            llm=self.llm,  
            chain_type="stuff",  
            retriever=self.vectorstore.as_retriever()  
        ) 

    def query(self,query):
        return self.qa(query)
    
