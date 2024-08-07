from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import OpenAIEmbeddings
###import load_examples as ex
import os

def load_examples(source_service,target_service):
    input_code = ""
    output_code = ""
    example_dir = r"C:\Users\vivprasad\Desktop\Mine\DataScienceCourse\LangChainPractice\FewShotExample\data\examples"

    for filename in os.listdir(example_dir):
        if source_service in filename:
            with open(os.path.join(example_dir,filename)) as in_file:
                input_code = in_file.read()
        if target_service in filename:
            with open(os.path.join(example_dir,filename)) as out_file:
                output_code = out_file.read()
    return input_code,output_code

example_template = """
User: {request}
input_code: {input_code}
output_code: {output_code}
"""
prefix = """
You are a code converter tool which converts code for a specific service from one cloud platform to a service on another cloud platform.
You have received request that contains the input code, the service name, the source cloud platform, the target service and cloud platform.
Use the given pieces of context to convert the code and generate output code for target service on target cloud platform.
If you don't understant the context look for relevant information from your knowlegde base. Do not return an empty string ("").
"""

suffix = """
User: {request}
input_code: {input_code}
AI:
"""
example_prompt = PromptTemplate(
    input_variables=["request","input_code","output_code"],
    template=example_template
)
#print(example_prompt)

def exampleTemplate(request,source_service,target_service):
    input_code,output_code = load_examples(source_service,target_service)
    examples = {
            "request": request,
            "input_code": f'{input_code!a}',
            "output_code": f'{output_code!a}'
        }

    few_shot_prompt_template = FewShotPromptTemplate(
        examples=[examples],
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["request","input_code"],  #These variables are used in the prefix and suffix
    )

    return few_shot_prompt_template