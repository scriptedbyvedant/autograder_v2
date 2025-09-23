
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
| 🚀 **MLOps & LMS Ready** | Designed with versioned feedback for CI/CD and MLOps workflows (MLflow, GitHub Actions), with planned integration for LMS like ILIAS. |
| 🖼️ **Multimodal Support** | Architected to support future evaluation of image-based answers, handwritten notes, and scientific sketches using OCR technology. |

</div>

---

### 🛠️ Technology Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama" />
</p>

---

## 🚀 Getting Started

This guide will walk you through deploying the AI Grading Framework on your local machine.

### 1. Prerequisites

- Python 3.8+ | Docker | PostgreSQL | [Ollama](https://ollama.com/)

### 2. Environment Setup

**Step 1: Clone the Repository**
```bash
git clone https://github.com/vedant-m/multi-agent-llm-grader.git
cd multi-agent-llm-grader
```

**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Configure the Database**

Log in to PostgreSQL and create the database and user. 
```sql
CREATE DATABASE autograder_db;
CREATE USER vedant WITH PASSWORD 'vedant';
GRANT ALL PRIVILEGES ON DATABASE autograder_db TO vedant;
```
> **Note:** The application uses the `credentials.yaml` file for database settings. If you use different credentials, please update this file accordingly.

**Step 4: Initialize the Database Schema**
```bash
python init_db.py
```

**Step 5: Pull the Local LLM**

Ensure Ollama is running, then pull the required `mistral` model.
```bash
ollama pull mistral
```

### 3. Running the Application

Ensure the **Docker** daemon and the **Ollama** application are running, then launch the app:
```bash
streamlit run app.py
```
Your browser will open to `http://localhost:8501`. You can now log in and begin exploring!

---

### 🗺️ Project Roadmap

- [ ] **Full MLOps Integration:** Implement an end-to-end MLflow pipeline for tracking experiments and versioning models/feedback.
- [ ] **LMS Integration:** Develop production-ready connectors for popular Learning Management Systems like ILIAS and Canvas.
- [ ] **Activate Multimodal Grading:** Implement the OCR and image-processing pipeline for grading graphical and handwritten submissions.
- [ ] **Analytics Dashboard:** Build a dashboard for educators to visualize class performance, question difficulty, and grading consistency over time.

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📜 License

Distributed under the **GNU General Public License v3.0**. See `LICENSE` for more information.
