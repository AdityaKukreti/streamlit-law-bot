from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LawBot:

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
        self.loader = DirectoryLoader('Database/')
        self.docs = self.loader.load()

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = self.text_splitter.split_documents(self.docs)
        self.vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings()) 
        self.retriever = self.vectorstore.as_retriever()


# 2. Incorporate the retriever into a question-answering chain.
        self.system_prompt = (
            """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know.
            \n\n
            {context}"""
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
            ]
        )

        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, self.question_answer_chain)


    def getResponse(self,query):
        response = self.rag_chain.invoke({"input": query})
        return response["answer"]