
# 🤖 AI Grading Framework: A Multi-Agent Approach

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

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

### 🏗️ Project Architecture

The system is designed with a modern, multi-tiered architecture to ensure a clean separation of concerns:

- **Frontend:** A user-friendly interface built with **Streamlit** provides all user-facing interactions, from data upload to grade reviews.
- **Backend:** A robust **Python** server acts as the central orchestrator, handling business logic, database transactions, and job delegation to the AI engine.
- **Database:** A **PostgreSQL** database serves as the single source of truth, storing all user data, submissions, and grading results with relational integrity.
- **AI & Data Layer:** This core layer includes:
  - The **AI Grading Engine** which manages the multi-agent system.
  - **Ollama** for serving local LLMs like Mistral.
  - A **FAISS Vector Store** that powers the RAG system's memory.
  - A **Docker Sandbox** for securely executing and testing student code.

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

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📜 License

Distributed under the **GNU General Public License v3.0**. See `LICENSE` for more information.
