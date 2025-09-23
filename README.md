
# 🤖 AI Grading Framework: A Multi-Agent Approach

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, AI-powered grading platform designed to bring reliability, consistency, and transparency to automated academic assessment. This project moves beyond single-model limitations by implementing a sophisticated multi-agent architecture where a team of AI agents collaborates to grade student work. 

Featuring a full Human-in-the-Loop (HITL) workflow, secure sandboxed code execution, and a RAG-powered institutional memory, this framework is built for the future of grading.

---

### ✨ Core Features

<div align="center">

| Feature | Description |
| :---: | :--- |
| 🤖 **Multi-Agent Collaboration** | Simulates a peer review by using diverse AI agents to grade concurrently, ensuring fairer, more robust, and less biased scoring through consensus. |
| 💡 **Explainable AI** | Delivers transparent, rubric-aligned justifications for every score. Understand not just the *what*, but the *why* behind each grade. |
| 🧠 **RAG-Powered Consistency** | Leverages a FAISS vector store to build an institutional memory from human-verified corrections, ensuring consistent application of standards over time. |
| 🔒 **Secure Code Evaluation** | Executes programming assignments in an isolated Docker sandbox, combining objective `unittest` results with qualitative AI feedback on code style. |
| 🧑‍🏫 **Human-in-the-Loop** | Provides educators with an intuitive UI to review, edit, and finalize all AI-generated grades, ensuring they always have the final say. |

</div>

---

## 🚀 Getting Started

This guide will walk you through deploying the AI Grading Framework on your local machine.

### 1. Prerequisites

Before you begin, ensure the following services are installed and running:

- Python 3.8+
- Docker
- PostgreSQL
- [Ollama](https://ollama.com/) for serving local LLMs

### 2. Environment Setup

Follow these steps to configure your environment and install the necessary components.

**Step 1: Clone the Repository**

Open your terminal and clone the project source code.
```bash
git clone https://github.com/vedant-m/multi-agent-llm-grader.git
cd multi-agent-llm-grader
```

**Step 2: Install Dependencies**

Create a virtual environment (optional but recommended) and install the required Python packages.
```bash
pip install -r requirements.txt
```

**Step 3: Configure the Database**

Log in to PostgreSQL and create the dedicated user and database for the application. These credentials match the default `credentials.yaml`.
```sql
-- Connect via psql or your preferred SQL client
CREATE DATABASE autograder_db;
CREATE USER vedant WITH PASSWORD 'vedant';
GRANT ALL PRIVILEGES ON DATABASE autograder_db TO vedant;
```

**Step 4: Initialize the Database Schema**

Run the initialization script to automatically set up all required tables.
```bash
python init_db.py
```

**Step 5: Pull the Local LLM**

Ensure Ollama is running, then pull the `mistral` model, which is used by the grading engine.
```bash
ollama pull mistral
```

### 3. Running the Application

With the environment configured, you are now ready to launch the application.

**Step 1: Start Services**

Ensure the **Docker** daemon and the **Ollama** application are running in the background.

**Step 2: Launch the Streamlit App**

From the root of the project directory, run the following command:
```bash
streamlit run app.py
```

Your web browser will automatically open to the application's local URL (usually `http://localhost:8501`). You can now log in and begin exploring the future of grading!

## 📘 Usage Example: Professor PDF Format

To ensure accurate parsing, professor documents containing the rubric and questions should follow a clear format. The system is designed to parse key-value pairs and section headers.

<div style="background: #f9fafb; border-radius: 8px; padding: 18px; font-family: monospace; font-size: 15px; color: #234; margin-bottom: 18px; margin-top: 2px; border-left: 4px solid #477ddb; box-shadow: 0 1.5px 7px #253d6a12; overflow-x: auto; white-space: pre-wrap;">
Professor: Dr. Smith<br>
Course: AI Fundamentals<br>
Assignment No: 2<br><br>

Q1:<br>
Question: Explain supervised vs. unsupervised learning.<br>
Ideal Answer: Supervised learning uses labeled data...<br>
Rubric:<br>
- Correct definition (2 pts)<br>
- Mention of labeled vs. unlabeled data (3 pts)
</div>
