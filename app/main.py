import os
import pdfplumber
import faiss
import numpy as np
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Retrieve OpenAI API Key securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Error: OpenAI API Key not found. Please check your .env file.")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text.strip()

# Function to split text into smaller chunks
def split_text_into_chunks(text, chunk_size=500, overlap=50):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )
    return text_splitter.split_text(text)

# Function to generate embeddings and store in FAISS
def create_faiss_index(text_chunks, embedding_model):
    embeddings = embedding_model.embed_documents(text_chunks)
    dimension = len(embeddings[0])
    
    index = faiss.IndexFlatL2(dimension)
    np_embeddings = np.array(embeddings, dtype=np.float32)
    index.add(np_embeddings)

    return index, text_chunks

# Function to perform question answering using GPT
def ask_question(question, texts, embedding_model):
    db = LangchainFAISS.from_texts(texts, embedding_model)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    result = qa_chain.invoke(question)
    return result

# Main execution
if __name__ == "__main__":
    pdf_path = "data/lawsofmotion.pdf"  # Use your actual file name here

    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found!")
    else:
        extracted_text = extract_text_from_pdf(pdf_path)
        print("Extracted Text Preview:", extracted_text[:500])
        text_chunks = split_text_into_chunks(extracted_text)
        print(f"Total Chunks Created: {len(text_chunks)}")

        embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        index, chunk_data = create_faiss_index(text_chunks, embedding_model)
        print("FAISS index created successfully!")

        # Update your question here
        question = "What is Newton’s third law of motion?"
        answer = ask_question(question, chunk_data, embedding_model)
        print(f"\nQ: {question}\nA: {answer}")
