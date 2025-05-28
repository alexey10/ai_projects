#Loading Documents Using LangChain for Different Sources


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from pprint import pprint
import json
from pathlib import Path
import os
import nltk
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredFileLoader

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


#Loading from TXT files
os.system("wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Ec5f3KYU1CpbKRp1whFLZw/new-Policies.txt")

loader = TextLoader("new-Policies.txt")
loader

data = loader.load()

data
pprint(data[0].page_content[:1000])

#Loading from PDF files

pdf_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Q81D33CdRLK6LswuQrANQQ/instructlab.pdf"

loader = PyPDFLoader(pdf_url)

pages = loader.load_and_split()

print(pages[0])

for p,page in enumerate(pages[0:3]):
    print(f"page number {p+1}")
    print(page)


#PyMuPDFLoader

loader = PyMuPDFLoader(pdf_url)
loader
data = loader.load()
print(data[0])

#Loading from Markdown files

os.system("wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/eMSP5vJjj9yOfAacLZRWsg/markdown-sample.md")

markdown_path = "markdown-sample.md"
loader = UnstructuredMarkdownLoader(markdown_path)
loader

data = loader.load()

pprint(data)

#Loading from JSON file
os.system("wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hAmzVJeOUAMHzmhUHNdAUg/facebook-chat.json")

file_path='facebook-chat.json'
data = json.loads(Path(file_path).read_text())

pprint(data)

loader = JSONLoader(
    file_path=file_path,
    jq_schema='.messages[].content',
    text_content=False)

data = loader.load()

pprint(data)

#Loading from CSV file
os.system("wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IygVG_j0M87BM4Z0zFsBMA/mlb-teams-2012.csv")
loader = CSVLoader(file_path='mlb-teams-2012.csv')
data = loader.load()

data

loader = UnstructuredCSVLoader(
    file_path="mlb-teams-2012.csv", mode="elements"
)
data = loader.load()

data[0].page_content
print(data[0].metadata["text_as_html"])

import requests
from bs4 import BeautifulSoup

url = 'https://www.ibm.com/topics/langchain'
response = requests.get(url)

soup = BeautifulSoup(response.content, 'html.parser')
print(soup.prettify())

loader = WebBaseLoader("https://www.ibm.com/topics/langchain")
data = loader.load()
data

loader = WebBaseLoader(["https://www.ibm.com/topics/langchain", "https://www.redhat.com/en/topics/ai/what-is-instructlab"])
data = loader.load()
data

#Loading from Word file

os.system("wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/94hiHUNLZdb0bLMkrCh79g/file-sample.docx")
loader = Docx2txtLoader("file-sample.docx")
data = loader.load()
data


files = ["markdown-sample.md", "new-Policies.txt"]
loader = UnstructuredFileLoader(files)
data = loader.load()
data

from langchain_community.document_loaders import PyPDFium2Loader

loader = PyPDFium2Loader("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Q81D33CdRLK6LswuQrANQQ/instructlab.pdf")

data = loader.load()
print(data)
