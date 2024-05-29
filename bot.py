# from langchain_openai import ChatOpenAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_community.document_loaders import DirectoryLoader
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import pinecone
# import os

# class LawBot:

#     def __init__(self):
#         self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
#         self.loader = DirectoryLoader('Database/')
#         self.docs = self.loader.load()

#         self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         self.splits = self.text_splitter.split_documents(self.docs)

#         # Initialize Pinecone
#         # pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment="us-west1-gcp")
#         self.pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

#         # Create Pinecone index
#         index_name = "lawbot-index"
#         # if index_name not in self.pc.list_indexes():
#         #     self.pc.create_index(index_name,dimension=1536,spec=pinecone.ServerlessSpec(cloud = "aws",region="us-east-1"))
        
#         self.index = self.pc.Index(index_name)

#         # Convert documents to embeddings and store in Pinecone
#         self.embeddings = OpenAIEmbeddings()
#         # vectors = [(str(i), self.embeddings.embed_documents(split.page_content)) for i, split in enumerate(self.splits)]
#         vectors = []
#         for i,split in enumerate(self.splits):
#             # print(split.page_content)
#             vectors.append(self.embeddings.embed_query(split.page_content))
#         print(vectors)
#         # self.index.upsert(vectors)

#         # # Create retriever
#         # self.retriever = self.index.as_retriever()

#         # # 2. Incorporate the retriever into a question-answering chain.
#         # self.system_prompt = (
#         #     """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know.
#         #     \n\n
#         #     {context}"""
#         # )

#         # self.prompt = ChatPromptTemplate.from_messages(
#         #     [
#         #         ("system", self.system_prompt),
#         #         ("human", "{input}"),
#         #     ]
#         # )

#         # self.question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
#         # self.rag_chain = create_retrieval_chain(self.retriever, self.question_answer_chain)

#     def getResponse(self, query):
#         response = self.rag_chain.invoke({"input": query})
#         return response["answer"]

# # Ensure to replace 'YOUR_PINECONE_API_KEY' with your actual Pinecone API key.

# # print(LawBot().getResponse("Deduction in advance payment"))
# LawBot()



import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
import pinecone

index_name = "lawbot-index"
embeddings = OpenAIEmbeddings()

# path to an example text file
loader = DirectoryLoader("Database/")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

vectorstore_from_docs = PineconeVectorStore.from_documents(
    docs,
    index_name=index_name,
    embedding=embeddings
)

# vectorstore = PineconeVectorStore(pinecone.Pinecone().Index(index_name),embeddings,"text")
# query = "What is Tax deduction in advance payment?"
# print(vectorstore.similarity_search(query,k=2))