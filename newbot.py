import os
import pinecone
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


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

   
        self.model = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
      

    def query(self,query):

        ans = [i for i in self.vectorstore.search(query,search_type="similarity")]

        relevance = self.model.chat.completions.create(model = "gpt-4o", 
                                                       messages=[
                                                           {
                                                               "role":"system",
                                                               "content":"""Given a query and a big chunk of data, if the data chunk is relevant to the query at all or if its a query related to financial laws, return me "True" otherwise "False"."""
                                                            },
                                                            {
                                                                "role":"user",
                                                                "content":f"query: {query}\n\ndata: {ans}"
                                                            }
                                                        ],
                                                        stream=False
                                                    ).choices[0].message.content
        
        if (eval(relevance)):
            return self.model.chat.completions.create(
                model = "gpt-4o",messages = [
                    {
                        "role":"system",
                        "content":"""
                        Your job is to answer the user query using the context provided by adhering to the following rules:
                        1. Quote all provisions of Income Tax act present from the context, that are relevant to the answer.
                        2. Include all sections and amendments from context that are relevant to the answer.
                        3. Do not add any additional text from your end. 
                        4. Embed the text in appropriate html tags except html, head, body tags. For heading, don't use h1 and h2. Close all the content in the p tag.
                        """
                    },
                    {
                    "role":"user","content":f"""query: {query}\n\ntext_data:{ans}"""
                    }
                    ],
                    stream = False).choices[0].message.content
        else:
            return "I don't know. Perhaps try rephrasing your question."

    
    
