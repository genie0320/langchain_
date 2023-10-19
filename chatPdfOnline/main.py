from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")


# Load file
loader = PyPDFLoader("sample.pdf")
pages = loader.load_and_split()
# print(pages[0])

# Text Transform (Split file)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    # length_function=len,
    # add_start_index=True,
)

texts = text_splitter.split_documents(pages)
# print(texts[0])
# 위에서 .load_and_split()로 쪼갰는데, 왜 또 .split_documents(pages)가 필요한거지?

# print(texts)
# Embedding
embeddings_model = OpenAIEmbeddings()

# What is this?
# embeddings = embeddings_model.embed_documents(texts)
# print(len(embeddings), len(embeddings[0]))

# Store to Chromadb
db = Chroma.from_documents(texts, embeddings_model)


# Query function
question = "사업금액은?"
llm = ChatOpenAI(temperature=0, max_tokens=100)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(), llm=llm
)
answer = retriever_from_llm.get_relevant_documents(query=question)
# Coding front end.
print(answer, "-", len(answer))
print('Done')
