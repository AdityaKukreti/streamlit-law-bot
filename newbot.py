import os
import pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI  
# from langchain.chains import RetrievalQA,RetrievalQAWithSourcesChain
# from langchain.chains import create_retrieval_chain
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain


class LawBot:

    def __init__(self):
        self.pc = pinecone.Pinecone(api_key=os.environ['PINECONE_API_KEY'])
        self.spec = pinecone.ServerlessSpec(cloud = "aws", region = 'us-east-1')

        self.index_name = 'lawbot-index'

        
        self.index = self.pc.Index(self.index_name)

        self.model_name = "text-embedding-ada-002"
        self.embeddings = OpenAIEmbeddings(model=self.model_name,openai_api_key = os.environ['OPENAI_API_KEY'])

        self.vectorstore = PineconeVectorStore(  
            self.index, self.embeddings,'text' 
        )  

        self.system_prompt = (
    """You are an assistant for question-answering finance related questions only. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Any irrelevent answers wont be tolerated.
    
    \n\n
    {context}"""
)
        
        self.prompt = ChatPromptTemplate.from_messages(
    [
        ("system", self.system_prompt),
        ("human", "{input}"),
    ]
)

        self.llm = ChatOpenAI(  
            openai_api_key=os.environ['OPENAI_API_KEY'],  
            model_name='gpt-4o',  
            temperature=0.0  
        )  
        # self.qa = RetrievalQAWithSourcesChain.from_chain_type(  
        #     llm=self.llm,  
        #     chain_type="stuff",  
        #     retriever=self.vectorstore.as_retriever()  
        # ) 

        # self.question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.model = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        # self.rag_chain = create_retrieval_chain(self.vectorstore.as_retriever(k = 1), self.question_answer_chain)

    def query(self,query):
        # return self.rag_chain.invoke({"input": query})['context']
        ans = [i.page_content for i in self.vectorstore.search(query,search_type="similarity")]
        a = self.model.chat.completions.create(model = "gpt-4o",messages = [{"role":"system","content":"""given a big chunk of text and a query, your task is to determine whehter the given chunk is related to the query or not. If not, then say "I don't know.". Otherwise arrange the given chunk in a proper meaningful manner including a proper explanation of it at the end and return it. Do not add any additional text from your end. Embed the text in appropriate html tags except html, head, body tags. For heading, don't use h1 and h2."""},{"role":"user","content":f"""query: {query}\n\ntext_chunk:{ans}"""}],stream = False)
        return a.choices[0].message.content
    
