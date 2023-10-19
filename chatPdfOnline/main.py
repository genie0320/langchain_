import openai
import os
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
import streamlit as st
import tempfile
import sys

# these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

st.title("PDF Genie")
st.write("---")

api_key = st.text_input('OPEN AI API key를 넣어주세요.', type="password")
openai.api_key = api_key
# Make a temp folder can store uploaded file.


def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load file and Split by "page".
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    print('file loaded')
    return pages


uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

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

    embeddings_model = OpenAIEmbeddings(openai_api_key=api_key)

    # ## What is this? Why It causes typeerror?
    # embeddings = embeddings_model.embed_documents(texts)
    # print(len(embeddings), len(embeddings[0]))

    db = Chroma.from_documents(texts, embeddings_model)

    # ------ After create db, then UI comes out.
    query = st.text_input('PDF분석이 완료되었습니다. 질문을 입력해주세요.')

    if st.button("궁금해", type="primary"):
        with st.spinner('Wait for it...'):
            myllm = ChatOpenAI(temperature=0, max_tokens=100,
                               openai_api_key=api_key)
            qa = RetrievalQA.from_chain_type(
                llm=myllm,
                # chain_type="stuff",
                retriever=db.as_retriever())
            answer = qa.run(query)
            st.write(answer)

    # Query
    # Search similar phrase from db and Toss it to the llm model to make a structured sentence.

    print('Done')
