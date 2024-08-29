import streamlit as st
import pandas as pd
import sqlite3
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import DirectoryLoader
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

###env_vars = {"<KeyNeededHere>"} Add Key here OPEN_AI_KEY varaible
    
for key, value in env_vars.items():
    os.environ[key] = value

embedding = OpenAIEmbeddings()
global rag_chain

# Get summary of SQLite DB
def create_connection(db_path):
    global conn
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    return conn


def create_table_ddl(conn):
    cursor = conn.cursor()
    query = "select name, sql from sqlite_master where type='table';"
    cursor.execute(query)
    ddl = cursor.fetchall()
    for item in ddl:
        name = item[0]
        file_name = name + "_ddl.txt"
        full_path = "./DDL/tables_txt_format/" + file_name
        with open(full_path, 'w') as file:
            file.write(item[1])


def create_view_ddl(conn):
    query = "select name, sql from sqlite_master where type='view';"
    cursor = conn.cursor()
    cursor.execute(query)
    ddl = cursor.fetchall()
    for item in ddl:
        name = item[0]
        file_name = name + "_ddl.txt"
        full_path = "./DDL/views_txt_format/" + file_name
        with open(full_path, 'w') as file:
            file.write(item[1])


def loading_docs():
    loader = DirectoryLoader("./DDL/", glob="**/*.txt", show_progress=True, use_multithreading=True)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    split_docs = text_splitter.split_documents(docs)
    return split_docs


def create_retriever():
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=200)  # Create smaller chunks
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Create larger chunks
    # Assigning Vector store
    vectorstore = Chroma(
        collection_name="DDL_info", embedding_function=embedding, persist_directory="embeddings\\"
    )
    # This is where our larger chunks are stored
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    docs = loading_docs()
    retriever.add_documents(docs, ids=None)
    return retriever


def create_template():
    template = """You are an assistant to create best documentations for the given SQL Table names or Views. Try to establish internal relationship between tables as well.
    The documentation should explain the purpose of the tables and the significance of each columns.
    Use the following pieces of retrieved context to answer the question. 
    If you don't get any relevant match, just say that no relevant match is found.
    Also try not to hallucinate

    Question: {question} 
    Context: {context} 
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt


def create_rag_chain():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    prompt = create_template()
    if os.path.isdir('embeddings'):
        db = Chroma(collection_name='DDL_info', persist_directory="./embeddings", embedding_function=embedding)
        retriever = db.as_retriever()
    else:
        retriever = create_retriever()
    global rag_chain
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    # return rag_chain


def generate_output(query):
    # query = "Give me the names of the candidates who are good in techincal skills like Python,SQL etc."
    return rag_chain.invoke(query)


def get_db_summary(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all the table names

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Get row counts for each tables

    summary = []
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        summary.append((table_name, row_count))

    create_view_ddl(conn)
    create_table_ddl(conn)
    create_rag_chain()
    return summary


# Streamlit app
def main():
    # st.title("SQLite Database Summary App")
    db_path = None
    st.set_page_config(layout="wide")
    st.header("DDL Chatbot")
    col1,col3,col2= st.columns([7,0.5,5],gap="medium")

    # Upload SQLite DB
    with st.sidebar:

        uploaded_file = st.file_uploader("Upload your SQLite Database", type=".db")
        if uploaded_file is not None:
            db_path = os.path.join("temp", uploaded_file.name)
            with open(db_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Get Summary
    if db_path is not None:
        with col2:
            if db_path:
                summary = get_db_summary(db_path)
                st.header("Database Summary")
                df_summary = pd.DataFrame(summary, columns=["Table Name", "Row Count"])
                st.dataframe(df_summary)

        with col3:
            st.write(" ")

        with col1:

            #################
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # React to user input
            if prompt := st.chat_input("Please ask questions related to loaded DB?"):
                # Display user message in chat message container
                st.chat_message("user").markdown(prompt)
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})

                response = f"Output : {rag_chain.invoke(prompt)}"
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == '__main__':
    main()


