from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter

index_name = "lawbot-index"
embeddings = OpenAIEmbeddings()

# path to an example text file
loader = DirectoryLoader("Database/")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

vectorstore_from_docs = PineconeVectorStore.from_documents(
    docs,
    index_name=index_name,
    embedding=embeddings
)

# vectorstore = PineconeVectorStore(pinecone.Pinecone().Index(index_name),embeddings,"text")
# query = "What is Tax deduction in advance payment?"
# print(vectorstore.similarity_search(query,k=2))