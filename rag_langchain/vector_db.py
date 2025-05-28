#Create and Configure a Vector Database to Store Document Embeddings

import os

os.system("wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/BYlUHaillwM8EUItaIytHQ/companypolicies.txt")

from langchain_community.document_loaders import TextLoader

loader = TextLoader("companypolicies.txt")
data = loader.load()

data

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)

chunks = text_splitter.split_documents(data)

print(len(chunks))

#Embedding model

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

#Vector store in Chroma DB

from langchain_community.vectorstores import Chroma

ids = [str(i) for i in range(0, len(chunks))]

vectordb = Chroma.from_documents(chunks, watsonx_embedding, ids=ids)

for i in range(3):
    print(vectordb._collection.get(ids=str(i)))

vectordb._collection.count()

query = "Email policy"
docs = vectordb.similarity_search(query)
docs

vectordb.similarity_search(query, k = 1)

#FIASS DB

from langchain_community.vectorstores import FAISS
faissdb = FAISS.from_documents(chunks, watsonx_embedding, ids=ids)
for i in range(3):
    print(faissdb.docstore.search(str(i)))

query = "Email policy"
docs = faissdb.similarity_search(query)
docs

#Managing vector store: Adding, updating, and deleting entries

#Add

text = "Instructlab is the best open source tool for fine-tuning a LLM."

from langchain_core.documents import Document

new_chunk =  Document(
    page_content=text,
    metadata={
        "source": "ibm.com",
        "page": 1
    }
)

new_chunks = [new_chunk]

print(vectordb._collection.get(ids=['215']))

vectordb.add_documents(
    new_chunks,
    ids=["215"]
)

vectordb._collection.count()
print(vectordb._collection.get(ids=['215']))

#Update

update_chunk =  Document(
    page_content="Instructlab is a perfect open source tool for fine-tuning a LLM.",
    metadata={
        "source": "ibm.com",
        "page": 1
    }
)

vectordb.update_document(
    '215',
    update_chunk,
)
print(vectordb._collection.get(ids=['215']))

#Delete
vectordb._collection.delete(ids=['215'])
print(vectordb._collection.get(ids=['215']))

query = "Smoking policy"
docs = vectordb.similarity_search(query)
print(docs)
