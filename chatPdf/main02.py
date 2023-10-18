from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os

load_dotenv()

st.title('ChatPDF')
st.write("---")

# 파일 업로드 UI
uploaded_file = st.file_uploader("분석하실 PDF를 선택해주세요.")
st.write("---")


def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    print('file loaded')
    return pages


# 업로드시 동작하는 코드
if uploaded_file is not None:
    # with st.spinner('PDF를 학습중입니다.'):
    page02 = pdf_to_document(uploaded_file)
    # st.success('Done!')

    # load PDF you want split
    # loader = PyPDFLoader("sample.pdf")
    # pages = loader.load_and_split()

    # Split the long text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        # add_start_index=True,
    )

    texts = text_splitter.split_documents(page02)

    embeddings_model = OpenAIEmbeddings()

    # db = Chroma.from_documents(texts, embeddings_model)

    # save results to disk
    db2 = Chroma.from_documents(texts, embeddings_model,
                                persist_directory="./chroma_db")

    st.header("학습이 완료되었습니다.")
    question = st.text_input("질문을 입력해주세요.")

    if st.button('알려줘'):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                         temperature=0, max_tokens=1000)
        # retriever_from_llm = MultiQueryRetriever.from_llm(
        #     retriever=db.as_retriever(), llm=llm
        # )
        # docs = retriever_from_llm.get_relevant_documents(query=question)

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=db2.as_retriever(),
            # chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        )

        result = qa_chain({"query": question})
        # docs = db2.similarity_search(question)
        # print(len(docs))
        print(result)
        st.write(result)
