from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

# Create a sample PDF file
def create_sample_pdf():
    pdf_path = "data/sample.pdf"
    os.makedirs("data", exist_ok=True)  # Ensure the 'data' folder exists

    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawString(100, 750, "Document Q&A System Sample PDF")
    c.drawString(100, 730, "This document contains text that will be processed using AI.")
    c.drawString(100, 710, "Here are some sample points:")
    c.drawString(120, 690, "- AI can extract and answer questions from PDFs.")
    c.drawString(120, 670, "- This PDF will be used to test document embedding and retrieval.")
    c.drawString(120, 650, "- The system uses FAISS for storing vector embeddings.")

    c.save()
    print(f"Sample PDF created successfully at {pdf_path}")

# Run the function
if __name__ == "__main__":
    create_sample_pdf()
