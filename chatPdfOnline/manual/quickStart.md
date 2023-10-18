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

Now, install OpenAI python library

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
                {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
                {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
            ]
        )

        print(completion.choices[0].message)

Then run it.

    python openai-test.py

After some moment you can get something like this.

    {
    "role": "assistant",
    "content": "In the realm of code's poetic flow,\nWhere programmers seek wisdom's glow,\nLies the essence of a looping dance,\nConcept grand, called recursive chance.\n\nImagine a tale, a story grand,\nWhere pages

    --- omitted ---

    the code's grand design,\nGuiding us through complexities, oh so fine.\n\nNow, my friend, as you code and create,\nRemember the tale, this lesson innate.\nWith recursion, problems you shall tame,\nIn the magical symphony of programming's flame."
    }

---

_Q. .env 파일을 통해 Key를 관리하는 방법은?_
