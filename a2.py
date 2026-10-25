import os
from click import prompt
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Load environment variables (Make sure you have OPENAI_API_KEY in a .env file)
# load_dotenv()
load_dotenv()

hf_token = os.getenv("HF_TOKEN")






### STEP1::DOCUMENT LOADER  
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"C:\Users\yrsau\Desktop\Pranathi\pranathiss_knowledge_base (1).pdf")
documents = loader.load()


### STEP2::CHUNKING STRATEGY

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

#### STEP3::VECTOR STORE CREATION

from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


#### STEP4::CREATE VECTOR STORE
from langchain_community.vectorstores import FAISS

vector_db = FAISS.from_documents(
    chunks,
    embedding
)



retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)





### STEP5::LOAD LLM

from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token=hf_token,
    temperature=0.3,
    max_new_tokens=512
)


# user_query = input("Enter your question: ")
# docs = retriever.invoke(user_query)
# 
# context = "\n\n".join(doc.page_content for doc in docs)
# 
# formatted_prompt = prompt.format(
#     context=context,
#     question=user_query
# )
# 
# response = llm.invoke(formatted_prompt)
# 
# print("\nAnswer:\n")
# print(response)

from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()

user_query = input("Ask a question: ")

docs = retriever.invoke(user_query)
context = "\n\n".join(doc.page_content for doc in docs)

answer = chain.invoke({
    "context": context,
    "question": user_query
})

print(answer)
