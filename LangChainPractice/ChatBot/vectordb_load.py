import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader,TextLoader,PyPDFDirectoryLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
#loader = DirectoryLoader('data/knowledge_base/', loader_cls=TextLoader)
loader = UnstructuredExcelLoader("data/knowledge_base/Refine_data.xlsx", mode="single")
docs = loader.load()

### Splitting the document into chunks and creating splits
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

embeddings=OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])

def create_db(document_splitted, embedding_model_instance):
    print('here')
    db=None
    try:
        #db = Chroma.from_documents(documents=document_splitted, embedding=embeddings,
        #                            persist_directory="./chroma_db")
        db = Chroma.from_documents(filter_complex_metadata(document_splitted), embeddings,
                                    persist_directory="./chroma_db")
    except Exception as error:
        print(error)
        raise error
    return db

db = create_db(splits, embeddings)