#In-Context Engineering and Prompt Templates

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from huggingface_hub import login
login(token="hf_token")

def llm_model(prompt_txt, params=None):
    model_id = 'mistralai/mixtral-8x7b-instruct-v01'

    default_params = {
        "max_new_tokens": 256,
        "min_new_tokens": 0,
        "temperature": 0.5,
        "top_p": 0.2,
        "top_k": 1
    }

    if params:
        default_params.update(params)

    parameters = {
        GenParams.MAX_NEW_TOKENS: default_params["max_new_tokens"],  # this controls the maximum number of tokens in the generated output
        GenParams.MIN_NEW_TOKENS: default_params["min_new_tokens"], # this controls the minimum number of tokens in the generated output
        GenParams.TEMPERATURE: default_params["temperature"], # this randomness or creativity of the model's responses
        GenParams.TOP_P: default_params["top_p"],
        GenParams.TOP_K: default_params["top_k"]
    }
    
    credentials = {
        "url": "https://us-south.ml.cloud.ibm.com"
    }
    
    project_id = "skills-network"
    
    model = Model(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id
    )
    
    mixtral_llm = WatsonxLLM(model=model)
    response  = mixtral_llm.invoke(prompt_txt)
    return response

GenParams().get_example_values()

params = {
    "max_new_tokens": 128,
    "min_new_tokens": 10,
    "temperature": 0.5,
    "top_p": 0.2,
    "top_k": 1
}

prompt = "The wind is"

response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")

prompt = """Classify the following statement as true or false: 
            'The Eiffel Tower is located in Berlin.'

            Answer:
"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")

#One-shot prompt

params = {
    "max_new_tokens": 20,
    "temperature": 0.1,
}

prompt = """Here is an example of translating a sentence from English to French:

            English: “How is the weather today?”
            French: “Comment est le temps aujourd'hui?”
            
            Now, translate the following sentence from English to French:
            
            English: “Where is the nearest supermarket?”
            
"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")

#Chain-of-thought (CoT) prompting

params = {
    "max_new_tokens": 512,
    "temperature": 0.5,
}

prompt = """Consider the problem: 'A store had 22 apples. They sold 15 apples today and got a new delivery of 8 apples. 
            How many apples are there now?’

            Break down each step of your calculation

"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")

params = {
    "max_new_tokens": 512,
}

prompt = """When I was 6, my sister was half of my age. Now I am 70, what age is my sister?

            Provide three independent calculations and explanations, then determine the most consistent result.

"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # or your custom checkpoint

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Define generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50)

# Wrap into LangChain LLM
from langchain.llms import HuggingFacePipeline
mixtral_llm = HuggingFacePipeline(pipeline=generator)


example_text = """
               Last week's book fair was a delightful gathering of authors and readers, featuring discussions and book signings.
               """

example_category = "Literature"

text = """
       The concert last night was an exhilarating experience with outstanding performances by all artists.
       """

categories = "Entertainment, Food and Dining, Technology, Literature, Music."

template = """
           Example:
           Text: {example_text}
           Category: {example_category}

           Now, classify the following text into one of the specified categories: {categories}
           
           Text: {text}
           
           Category:
           
           """
prompt = PromptTemplate.from_template(template)
output_key = "category"

llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm, output_key=output_key)
response = llm_chain.invoke(input = {"example_text": example_text, "example_category":example_category ,"categories": categories, "text":text})
print(response["category"])




