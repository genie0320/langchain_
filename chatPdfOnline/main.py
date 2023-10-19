from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA

import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load file and Split by "page".
loader = PyPDFLoader("sample.pdf")
pages = loader.load_and_split()

# Text Transform (Split by certain given settings)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    # length_function=len,
    # add_start_index=True,
)

texts = text_splitter.split_documents(pages)

# Embedding Store to Chromadb.
# Need to install [tiktoken] to use embedding.

embeddings_model = OpenAIEmbeddings()

# ## What is this? Why It causes typeerror?
# embeddings = embeddings_model.embed_documents(texts)
# print(len(embeddings), len(embeddings[0]))

db = Chroma.from_documents(texts, embeddings_model)

# Query
# Search similar phrase from db and Toss it to the llm model to make a structured sentence.

myllm = ChatOpenAI(temperature=0, max_tokens=100)
qa = RetrievalQA.from_chain_type(
    llm=myllm,
    # chain_type="stuff",
    retriever=db.as_retriever())

query = "사업금액은?"
answer = qa.run(query)
print(answer)
print('Done')

# Coding frontend.
