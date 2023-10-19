<!-- @format -->

Things I learned about LangChain

How to use Llama2 in free of charge.
Download 'GGML(Llma2 Lighter ver)'

https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML Put downloaded file in to same folder.

Install Ctransformer.

pip install ctransformers
import CTransformers
from langchain.llms import CTransformers

llm = CTransformers(
model="llama-2-7b...the file name you downloaded earlier...",
model_type="llama"
)
Caution : You can NOT use this on internet server.
Should use English because Llama is unfamiliar with Korean.
