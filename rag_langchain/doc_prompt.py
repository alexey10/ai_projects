#Placing Whole Document into Prompt and Asking the Model

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import os
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader
from langchain_ibm import WatsonxLLM

def llm_model(model_id):
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
        GenParams.TEMPERATURE: 0.5, # this randomness or creativity of the model's responses
    }
    
    credentials = {
        "url": "https://us-south.ml.cloud.ibm.com"
    }
    
    project_id = "skills-network"
    
    model = ModelInference(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id
    )
    
    llm = WatsonxLLM(watsonx_model = model)
    return llm

llama_llm = llm_model('meta-llama/llama-3-3-70b-instruct')
llama_llm.invoke("How are you?")

os.system("wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/d_ahNwb1L2duIxBR6RD63Q/state-of-the-union.txt")

loader = TextLoader("state-of-the-union.txt")
data = loader.load()

content = data[0].page_content

template = """According to the document content here 
            {content},
            answer this question 
            {question}.
            Do not try to make up the answer.
                
            YOUR RESPONSE:
"""

prompt_template = PromptTemplate(template=template, input_variables=['content', 'question'])
prompt_template 

mixtral_llm = llm_model('mistralai/mixtral-8x7b-instruct-v01')

query_chain = LLMChain(llm=mixtral_llm, prompt=prompt_template)
query = "It is in which year of our nation?"
response = query_chain.invoke(input={'content': content, 'question': query})
print(response['text'])

query_chain = LLMChain(llm=llama_llm, prompt=prompt_template)
query_chain 

query = "It is in which year of our nation?"
response = query_chain.invoke(input={'content': content, 'question': query})
print(response['text'])

model_name = 'ibm/granite-3-8b-instruct'
granite_llm = llm_model(model_name)
query_chain = LLMChain(llm=granite_llm, prompt=prompt_template)
query = "It is in which year of our nation?"
response = query_chain.invoke(input={'content': content, 'question': query})
print(f"Response from model [{model_name}]: {response['text']}")

#model_names = [
#    'ibm/granite-3-8b-instruct',
#    'meta-llama/llama-3-2-3b-instruct'
#    'mistralai/Mixtral-8x7B-Instruct-v0.1'
#]
#
#query = "It is in which year of our nation?"
#
#for model_name in model_names:
#    llm = llm_model(model_name)
#    query_chain = LLMChain(llm=llm, prompt=prompt_template)
#    response = query_chain.invoke(input={'content': content, 'question': query})
#    
#    print(f"Response from model [{model_name}]: {response['text']}")


