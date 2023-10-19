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
