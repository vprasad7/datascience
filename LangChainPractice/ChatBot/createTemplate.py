from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import OpenAIEmbeddings
###import load_examples as ex
import os

def promptTemplate():

    template = """You are chatbot that provides information regarding data visulization and reporting tools.\n
    Please refer the below context given and answer the questions sent by the user.\n
    Do not add extra content from outside.\n
    Provide links and URLS where ever needed.\n

    context: {context}
    user: {request}

    output:"""
    custom_rag_prompt = PromptTemplate.from_template(template)
    return custom_rag_prompt