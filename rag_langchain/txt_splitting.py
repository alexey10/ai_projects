#Apply Text Splitting Techniques to Enhance Model Responsiveness

import os


os.system("wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/YRYau14UJyh0DdiLDdzFcA/companypolicies.txt")

with open("companypolicies.txt") as f:
    companypolicies = f.read()

print(companypolicies)

from langchain_core.documents import Document
Document(page_content="""Python is an interpreted high-level general-purpose programming language. 
                        Python's design philosophy emphasizes code readability with its notable use of significant indentation.""",
         metadata={
             'my_document_id' : 234234,
             'my_document_source' : "About Python",
             'my_document_create_time' : 1680013019
         })

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
)

texts = text_splitter.split_text(companypolicies)

texts

print(len(texts))

texts = text_splitter.create_documents([companypolicies], metadatas=[{"document":"Company Policies"}])  # pass the metadata as well

texts[0]

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)

texts = text_splitter.create_documents([companypolicies])

texts

print(len(texts))

from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

[e.value for e in Language]

RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)

PYTHON_CODE = """
    def hello_world():
        print("Hello, World!")
    
    # Call the function
    hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
python_docs = python_splitter.create_documents([PYTHON_CODE])
python_docs

RecursiveCharacterTextSplitter.get_separators_for_language(Language.JS)

JS_CODE = """
    function helloWorld() {
      console.log("Hello, World!");
    }
    
    // Call the function
    helloWorld();
"""

js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, chunk_size=60, chunk_overlap=0
)
js_docs = js_splitter.create_documents([JS_CODE])
print(js_docs)

from langchain.text_splitter import MarkdownHeaderTextSplitter

md = "# Foo\n\n## Bar\n\nHi this is Jim\n\nHi this is Joe\n\n### Boo \n\nHi this is Lance \n\n## Baz\n\nHi this is Molly"

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(md)
print(md_header_splits)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
)
texts = text_splitter.split_text(companypolicies)
print(texts)




