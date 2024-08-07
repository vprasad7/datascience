from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

def generate_stepback(question,llm):
    system = """You are a code conversion tool that converts code for a specific service from one cloud platform to another service on a different cloud platform.
    You will be asked to convert source cloud service code to target cloud service code.
    Given a specific user request to convert the code, write a more generic question that needs to be answered in order to know the specific details are needed for conversion.
    If you don't recognize a word or acronym to not try to rewrite it.
    Write concise questions."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    step_back = prompt | llm | StrOutputParser()
    return step_back

#stepback = sb.generate_stepback(request,llm)
#stepback_question = stepback.invoke({"question": request})
#print(f'stepback_question {stepback_question}')