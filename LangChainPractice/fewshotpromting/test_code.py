import pandas as pd
import numpy as np
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, DataFrameLoader
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
import os
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

model_name = 'AIMH/mental-longformer-base-4096'
model_kwargs = {'device':'cuda'}
encode_kwargs = {'normalize_embeddings':False}

embedding= HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)
document_path = "/content/drive/MyDrive/Colab_Notebooks/papers"
indicators = '''
"An overwhelming sense that one can't escape their current situation or problems."
"Alcohol or other substance use"
"Disconnection from friends, family, and social activities."
"Believing that nothing will ever get better or change."
'''

# to df
indicators = pd.DataFrame(indicators.split('\n'), columns=['indicators'])
# load document
loader = PyPDFDirectoryLoader(document_path)
documents = loader.load()

# make indicators a Document and append to document_splitted

df_loader = DataFrameLoader(indicators, page_content_column="indicators")
documents.extend(df_loader.load())

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)

#db=model_vectorstore.from_texts(content, embedding_model_instance, metadata)

def create_db(document_splitted, embedding_model_instance):

    model_vectorstore = FAISS
    db=None
    try:
        content = []
        metadata = []
        for d in document_splitted:
            content.append(d.page_content)
            metadata.append({'source': d.metadata})
        db=model_vectorstore.from_texts(content, embedding_model_instance, metadata)
    except Exception as error:
        print(error)
    return db

db = create_db(chunked_documents, embedding)
#store the db locally for future use
db.save_local('db.index')

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

model_path= "TheBloke/zephyr-7B-beta-AWQ"
task = "text-generation"
model_kwargs={
        "temperature": 0,
        "max_length": 512,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "num_return_sequences": 1
    }
pipeline_kwargs={
        "repetition_penalty":1.1
    }

from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_quantized(model_path, fuse_layer=True,trust_remote_code = False, safetensors = True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code = False)
pipe = pipeline(
    model=model,
    tokenizer=tokenizer,
    device="cuda",
    task=task
)
llm = HuggingFacePipeline(pipeline = pipe, model_kwargs=model_kwargs, pipeline_kwargs=pipeline_kwargs)

post1 = '''
my roomate drives me crazy. she bullies me and says horrible things. i am a very anxious person so i just hide in my room all day. i have not spoken to my family in weeks and have lost 10 pounds. what to do.
'''
post1_label = 'severe'

userid_1 = 1

post1_evidence = ["says horrible things",
                  "I'm a very anxious person",
                  "she bullies"]

post2 = '''
I think i am depressed, i do feel like eating or going to the gym. what do i do?
'''
post2_label = 'moderate'
userid_2 = 1

examples = [
    {
        "post": post1,
        "evidence": post1_evidence
    }
]

example_template = """
{context}
###POST: {question}
###EVIDENCE: {evidence}
"""
prefix = """
You are an expert psychologist.
You have a received information that the post's author is in one of 'Severe','Moderate',or 'Low' risk of depression.
Use the following pieces of context to select the spans of text that provide evidence of the risk level.
If you don't know the answer return an empty string (""). Do not make up an answer.

"""

suffix = """
{context}
###POST: {question}
###EVIDENCE:
"""
example_prompt = PromptTemplate(
    input_variables=["context","question", "evidence"],
    template=example_template
)
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["context","question"],#These variables are used in the prefix and suffix
    example_separator="\n\n"
)
def gen_resp(retriever, question):
  rag_custom_prompt = few_shot_prompt_template
  context = "\n".join(doc.page_content for doc in retriever.get_relevant_documents(query = question))
  rag_chain = (
      {"context": lambda x: context, "question": RunnablePassthrough()} | 
      rag_custom_prompt | 
      llm

  )
  answer = rag_chain.invoke(question)
  return answer

gen_resp(retriever, post2)