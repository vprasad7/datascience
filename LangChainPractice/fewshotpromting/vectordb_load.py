import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader,TextLoader,PyPDFDirectoryLoader
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
#loader = DirectoryLoader('data/knowledge_base/', loader_cls=TextLoader)
loader = PyPDFDirectoryLoader('data/knowledge_base/')
docs = loader.load()

### Splitting the document into chunks and creating splits
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

embeddings=OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])

def create_db(document_splitted, embedding_model_instance):
    print('here')
    db=None
    try:
        db = Chroma.from_documents(documents=document_splitted, embedding=embeddings,
                                    persist_directory="./chroma_db")
    except Exception as error:
        print(error)
    return db

db = create_db(splits, embeddings)