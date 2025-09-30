# Satark – Intelligent Document and Image Analysis System

## Project Overview
Satark is an AI-powered system designed to extract, process, and analyze content from multiple formats such as `.docx` files and images.  
The project leverages **OCR (Optical Character Recognition)**, **FAISS (Vector Database)**, and **Ollama LLMs** to enable efficient text extraction, chunking, semantic search, and intelligent question answering.

This solution is particularly useful for:
- Document intelligence  
- Compliance & auditing  
- Research analysis  
- Accessibility use cases  

---

## Installation Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/satark.git
cd satark
2. Create and activate a virtual environment
bash
Copy code
python -m venv venv
On Windows:

bash
Copy code
venv\Scripts\activate
On Mac/Linux:

bash
Copy code
source venv/bin/activate
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
4. Install and set up Ollama
Download Ollama from Ollama.ai

Pull the desired model (example: Mistral):

bash
Copy code
ollama pull mistral
5. Verify installation
bash
Copy code
python scripts/test_pipeline.py
Usage Guidelines
Place your .docx or image files inside the input/ folder.

Run the extraction script:

```bash
Copy code
python scripts/process.py
This will generate:

Extracted text → output/extracted_text.txt

Metadata → output/chunks_metadata.json

FAISS index → output/faiss_index.idx

Start the interactive query application:

bash
Copy code
python scripts/app.py
Ask natural language questions about your documents/images, for example:

"Summarize the document in 3 points"

"What are the key entities mentioned in the image text?"

Contributing to the Project
We welcome contributions to make Satark better.

Steps to contribute:

bash
Copy code
# Fork this repository
git checkout -b feature-name
git commit -m "Added feature-name"
git push origin feature-name
Then, open a Pull Request.

License Information
This project is licensed under the MIT License.
You are free to use, modify, and distribute the code with proper attribution.

Acknowledgments
Ollama → for providing local LLM support

Mistral, Llama, and Gemma models → for powering natural language understanding

FAISS → for efficient similarity search

pytesseract → for OCR-based text extraction

Open-source contributors and the research community → for continuous improvements

pgsql
Copy code

---







