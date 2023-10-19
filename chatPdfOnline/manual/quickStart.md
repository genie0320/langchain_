<!-- @format -->

# Quick Start OpenAI.

## 1. Install python ver. you want.

    https://www.python.org/downloads/

## 2. Setup a virtual env.

Type below in IDE's powershell (NOT os's)

    python -m venv openai-env

Then Activate it. (Important!)

    openai-env\Scripts\activate

    // In case you want deactivate...
    deactivate

Now, install OpenAI and python library

    pip install --upgrade openai

## 3. Setup API key for the project

This code is from chatGPT 3.5

    $env:OPENAI_API_KEY = "<API-KEY>"

Then verify it.

    echo $env:OPENAI_API_KEY

## 4. Verifing connection to OpenAI

Make a file nameed 'openai-test.py' and code like below.

    import os
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair, whithin 20words"},
                {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming, whithin 20words"}
            ]
        )

        print(completion.choices[0].message)

Then run it.

    python openai-test.py

After some moment you can get something like this.

    {
    "role": "assistant",
    "content": "In a program's embrace, recursion finds its place.\nIt's a dance where functions call their own selves, with grace."
    }

---

## 5. Now you can install all the things.

    1. Python SDK and Tokenizer
    pip install openai
    pip install tiktoken

    2. LLM or Chat model
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI

    3. Text embedding model
    from langchain.embeddings import OpenAIEmbeddings
