 
# 📘 Document_QnA_GPT: AI-Powered PDF Q&A Assistant

A production-ready GenAI application that allows users to upload any PDF document (e.g., academic material, HR policies, legal papers) and interact with it through natural language queries. Built with OpenAI's GPT, LangChain, FAISS, and Streamlit, the system retrieves and answers questions based on document content in real-time.


---

## 🚀 Features

- 🧾 Extracts meaningful content from PDF files
- 🔍 Performs semantic search over document chunks using FAISS
- 🤖 Answers questions using Retrieval-Augmented Generation (RAG) with OpenAI GPT
- 📊 User-friendly Streamlit UI for fast interaction
- 🔐 Securely handles API keys via `.env` integration
- 🧩 Modular, extensible architecture (ready for APIs, multi-doc search, etc.)


---

## 📁 Folder Structure
Document_QnA_GPT/ ├── app/ # Core application logic (Streamlit, main pipeline) │ ├── main.py # RAG pipeline: extract, embed, answer │ └── streamlit_app.py # Streamlit frontend interface │ ├── data/ # PDF documents for testing (excluded via .gitignore) │ ├── utils/ # Helper scripts (PDF generation, future preprocessing) │ └── generate_sample_pdf.py │ ├── .gitignore # Ignored files/folders ├── requirements.txt # Python dependencies ├── README.md # You're reading it :) └── .env # OpenAI key (not committed)

## 🧠 Use Cases

- 👨‍🎓 NEET/NCERT Chapter Assistants (Physics, Bio, Chem)
- 🏢 HR Policy Q&A
- 📑 Legal Document Understanding
- 💰 Finance & Compliance Review
- 📚 Research Paper Summarization & Analysis

---

## 🛠 Tech Stack

| Tool/Library      | Purpose                         |
|-------------------|----------------------------------|
| `OpenAI GPT-4`    | Natural language understanding  |
| `LangChain`       | Retrieval-Augmented Generation  |
| `FAISS`           | Vector similarity search        |
| `pdfplumber`      | PDF parsing                     |
| `Streamlit`       | Interactive UI                  |
| `Python`          | Backend development             |

---

## 📸 Screenshot

> _User uploads a NEET Physics chapter (PDF) and asks: "What is inertia?"_

![PDF Q&A Screenshot](app/static/screenshot.png)

---

## 🧪 How to Run Locally

### 🔹 Step 1: Clone the repository


git clone https://github.com/himanshu-dandle/Document_QnA_GPT.git
cd Document_QnA_GPT



### 🔹 Step 2: Set up virtual environment

	python -m venv venv
	venv\Scripts\activate         # On Windows
	pip install -r requirements.txt



###🔹 Step 3: Add OpenAI Key

	Create a .env file:
	OPENAI_API_KEY=your-openai-key-here
	

###🔹 Step 4: Run the app
	streamlit run app/streamlit_app.py

## 🌍 Live Demo


>

###🔹🔮 Future Work
 💬 Add conversational memory (chat history)

 📁 Support multiple PDF uploads and indexing

 🧪 Add unit testing and error handling

 🌐 REST API version using FastAPI for integration

 ☁️ One-click deploy on Streamlit Cloud or Hugging Face Spaces

 🔒 Role-based access and document privacy controls


)

👨‍💻 Author
Himanshu Dandle
📧 Email : himanshu.dandle@gmail.com

