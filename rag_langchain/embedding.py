#Embeding documents using watsonx's embedding model

import os

os.system("wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/i5V3ACEyz6hnYpVq6MTSvg/state-of-the-union.txt")

from langchain_community.document_loaders import TextLoader

loader = TextLoader("state-of-the-union.txt")
data = loader.load()

data

#Splitting data

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)
chunks = text_splitter.split_text(data[0].page_content)
print(len(chunks))

chunks

#Building model

from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings

embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}

watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=embed_params,
)

#Querry embeddings


query = "How are you?"

query_result = watsonx_embedding.embed_query(query)

print(len(query_result))

print(query_result[:5])

#Document embeddings

doc_result = watsonx_embedding.embed_documents(chunks)
print(len(doc_result))
doc_result[0][:5]
print(len(doc_result[0]))


#Build model
from langchain_community.embeddings import HuggingFaceEmbeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
huggingface_embedding = HuggingFaceEmbeddings(model_name=model_name)

#Query embeddings
query = "How are you?"
query_result = huggingface_embedding.embed_query(query)
query_result[:5]

#Document embeddings

doc_result = huggingface_embedding.embed_documents(chunks)
doc_result[0][:5]
print(len(doc_result[0]))

#Using another watson model

from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings

embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}

watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-30m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=embed_params,
)

doc_result = watsonx_embedding.embed_documents(chunks)

print(doc_result[0][:5])


