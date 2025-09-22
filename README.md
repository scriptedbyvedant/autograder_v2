# LLM Academic Grading System

This project is an AI-powered multilingual grading platform designed to assist educators in evaluating student submissions. It leverages Large Language Models (LLMs) to provide automated grading, explainability, and rich analytics.

## Features

- **Automated Grading:** Utilizes LLMs to grade student answers based on a provided rubric.
- **Explainability:** Generates explanations for the assigned grades, providing insights into the grading process.
- **Multilingual Support:** Supports grading and feedback in multiple languages.
- **Rich Analytics:** Offers detailed analytics on student performance and grading consistency.

## Getting Started

### Prerequisites

- Python 3.8+
- Pip
- Ollama

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/llm-academic-grading-system.git
   cd llm-academic-grading-system
   ```

2. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the environment variables:**

   Create a `.env` file in the root directory and add the following:

   ```
   OLLAMA_MODEL=mistral
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   DB_TYPE=postgres
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=grading_db
   DB_USER=admin
   DB_PASSWORD=password
   ```

### Running the Application

1. **Start the Ollama service:**

   ```bash
   ollama serve
   ```

2. **Run the Streamlit application:**

   ```bash
   streamlit run main.py
   ```

## Usage

1. **Upload the grading rubric and student submissions.**
2. **The application will automatically grade the submissions and provide feedback.**
3. **Review the grading results and analytics.**

## Contributing

Contributions are welcome! Please open an issue or submit a pull request to contribute to this project.

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for more details. Your professor suggested the GPL License, so you may want to update the license file accordingly.

## Credits

This project is built upon the work of numerous open-source projects and concepts. We extend our gratitude to the developers and communities behind these indispensable tools.

### AI and Machine Learning

*   **Agentic AI and Multi-Agent Systems:** The core grading engine employs a multi-agent architecture, where different specialized agents handle various tasks like routing, grading different types of content (text, math, code), and fusing their outputs. This design is inspired by the principles of agent-based systems and makes the grading process more robust and modular.
*   **Retrieval-Augmented Generation (RAG):** We use a RAG pipeline to provide relevant context to the grading models. This involves a `SimpleVectorStore` inspired by vector databases like **FAISS** and **Chroma**.
*   **LLM Orchestration:** The concepts of chaining and agentic behavior are influenced by frameworks like **LangChain**.
*   **Large Language Models:** The system is designed to work with various LLMs, and we are thankful for the open-source models and the tools to run them, such as **Ollama**.
*   **Core Libraries:**
    *   **Hugging Face Transformers:** For access to state-of-the-art models.
    *   **Sentence-Transformers:** For generating high-quality embeddings for our RAG pipeline.
    *   **PyTorch:** As the foundational deep learning framework.
*   **Fine-Tuning:**
    *   **PEFT (Parameter-Efficient Fine-Tuning):** For efficient model fine-tuning.
    *   **bitsandbytes:** For 8-bit and 4-bit quantization.
    *   **TRL (Transformer Reinforcement Learning):** For fine-tuning transformer models.

### Backend and Data Processing

*   **PDF Parsing:** **PyMuPDF (Fitz)** is used for extracting text and other data from PDF documents.
*   **Database:** **PostgreSQL** serves as our robust and reliable database.
*   **Symbolic Mathematics:** **SymPy** is used for handling and evaluating mathematical expressions.
*   **Numerical Computing:** **NumPy** and **Pandas** are used for data manipulation and analysis.

### Frontend

*   **Web Framework:** The user interface is built with **Streamlit**, an open-source app framework for Machine Learning and Data Science projects.

### General

*   **Python:** The programming language this project is built in.
