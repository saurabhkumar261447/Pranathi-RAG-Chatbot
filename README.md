# Pranathi RAG Chatbot

A Python proof of concept for asking questions about a PDF. The script loads the PDF, splits its text into chunks, indexes the chunks with FAISS and Hugging Face sentence-transformer embeddings, retrieves relevant chunks, and sends them with the question to a Hugging Face-hosted language model.

## Project files

- `a2.py` — loads the PDF, builds the vector index, retrieves context, and generates an answer.
- `pranathiss_knowledge_base (1).pdf` — the PDF used as the knowledge base.
- `requirements.txt` — Python dependencies.
- `.env` — local environment variables; do not commit this file or put secrets in the README.

## Requirements

- Python 3.10 or newer
- A Hugging Face access token accepted by the configured model
- Internet access the first time the embedding model is downloaded and when calling the hosted language model

## Setup on Windows

Open PowerShell in the project directory and create a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies and the Hugging Face integration packages used by `a2.py`:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install langchain-huggingface sentence-transformers
```

Create a `.env` file in the project directory containing your Hugging Face token:

```dotenv
HF_TOKEN=your_hugging_face_token
```

Keep the real token private. Do not commit `.env` to GitHub. If the token has already been exposed, revoke it and create a replacement.

## Before running

Update the PDF path in `a2.py` if the project is not located at the path currently hard-coded in the script. The configured model is `mistralai/Mistral-7B-Instruct-v0.3`; make sure your Hugging Face account can access it.

The script also needs a LangChain prompt template for its `context` and `question` inputs. It currently imports `prompt` from Click, which is a command-line function and cannot be used as the LangChain prompt in the chain. Replace that import with:

```python
from langchain_core.prompts import PromptTemplate
```

Then, before `chain = prompt | llm | StrOutputParser()`, define the template:

```python
prompt = PromptTemplate.from_template(
    "Answer the question using the context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)
```

## Run

From the project directory, with the virtual environment activated:

```powershell
python a2.py
```

Enter a question when prompted. The script retrieves up to three relevant PDF chunks and prints the model's answer. The vector index is built in memory each time the script runs; it is not saved between runs.

## Notes

- `HF_TOKEN` is read from `.env` using `python-dotenv`.
- The embedding model is `sentence-transformers/all-MiniLM-L6-v2`.
- The language model is `mistralai/Mistral-7B-Instruct-v0.3`, called through Hugging Face.
- `requirements.txt` currently does not list `langchain-huggingface` or `sentence-transformers`; the setup commands install them separately.
