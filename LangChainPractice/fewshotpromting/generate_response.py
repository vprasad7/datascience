from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv ,find_dotenv
load_dotenv(dotenv_path = r'C:\Users\vivprasad\Desktop\Mine\DataScienceCourse\LangChainPractice\FewShotExample\.env')
find_dotenv()
import load_examples as ex
import createExampleTemplate as cext
import os
import traceback
import stepback as sb

def gen_resp(request):
  answer = ""
  try:
    few_shot_prompt = cext.exampleTemplate(request,'AWS_Lambda_Python','GCP_CloudFn_Python')
    #print(f'few_shot_prompt {few_shot_prompt}')
    prompt = few_shot_prompt.format(
      request=request,
      input_code=input_code
    )
    print(f'prompt {prompt}')
    context = "\n".join(doc.page_content for doc in retriever.get_relevant_documents(query = request))

    rag_chain = (
      {"context": lambda x: context, "request": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()

    )
    answer = rag_chain.invoke(request)
  except Exception as e:
    print(traceback.format_exc())
  return answer

input_code = '''
import boto3
ec2 = boto3.resource('ec2')
instance = ec2.create_instances(
    ImageId='ami-123456',
    MinCount=1,
    MaxCount=1,
    InstanceType='t2.micro'
)
print(instance[0].id)
'''

embeddings=OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
request = f'Convert the given AWS Lambda Python Function to GCP Cloud Function in Python'
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)
# load Context from Vector DB
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
stepback = sb.generate_stepback(request,llm)
stepback_question = stepback.invoke({"question": request})
print(f'stepback_question {stepback_question}')

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

output_code = gen_resp(request)
print(output_code)