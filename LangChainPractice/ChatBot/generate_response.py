from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv ,find_dotenv
import createTemplate as temp
import os
import traceback
load_dotenv()

#with open(r'C:\Users\vivprasad\Desktop\Mine\DataScienceCourse\LangChainPractice\Inter-Cloud_migration_accelerator\data\examples\AWS_Lambda_ELV.txt') as in_file:
#  input_code = in_file.read()

def gen_resp(request):

  embeddings=OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
  #request = 'what products are available in Power BI.'
  llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)
  # load Context from Vector DB
  db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

  retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

  answer = ""
  try:
    context = "\n".join(doc.page_content for doc in retriever.invoke(request))
    ###print(f'context {context}')
    prompt = temp.promptTemplate()
    
    rag_chain = (
      {
      "context": lambda x: context,
      "request": lambda x:request,"request": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()

    )
    answer = rag_chain.invoke(request)
  except Exception as e:
    print(traceback.format_exc())
  return answer

#answer = gen_resp()
#print(answer)