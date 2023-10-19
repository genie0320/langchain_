<!-- @format -->

# ChatPDF.

## OPEN_API_KEY setting

    // Set environmental variable (in virtual env., for this project)

    $env:OPENAI_API_KEY = "<API-KEY>"

    // Then verify it.

    echo $env:OPENAI_API_KEY

    // Lastly load it.
    import os
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")

## Load Pdf

    // install
    pip install langchain
    pip install pypdf

    // Use
    from langchain.document_loaders import PyPDFLoader

    loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
    pages = loader.load_and_split()

## Transform text to small byte.

    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True,
    )
    texts = text_splitter.split_documents(pages)

---

## Embed processed file

    from langchain.embeddings import OpenAIEmbeddings

    embeddings_model = OpenAIEmbeddings()

## Store it into Chromadb

    from langchain.vectorstores import Chroma

    db = Chroma.from_documents(texts, embeddings_model)

## Make a Chat function.

    from langchain.chat_models import ChatOpenAI
    from langchain.retrievers.multi_query import MultiQueryRetriever

    question = "What are the approaches to Task Decomposition?"
    llm = ChatOpenAI(temperature=0)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(), llm=llm
    )

# Work on

_Q. .env 파일을 통해 Key를 관리하는 방법은?_

---

# History

[Quick Start OpenAI.][manual/quickStart.md]
