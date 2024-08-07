import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_examples(source_service,target_service):
    input_code = ""
    output_code = ""
    example_dir = "data/examples"

    for filename in os.listdir(example_dir):
        if source_service in filename:
            with open(os.path.join(example_dir,filename)) as in_file:
                input_code = in_file.read()
        if target_service in filename:
            with open(os.path.join(example_dir,filename)) as out_file:
                output_code = out_file.read()

    examples = [
        {
            "request": f'''Convert the given AWS Lambda Python Function to GCP Cloud Function in Python.\n {input_code}''',
            "output_code": output_code
        }
    ]

    return examples