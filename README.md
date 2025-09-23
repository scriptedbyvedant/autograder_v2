
# 🤖 AI Grading Framework: A Multi-Agent Approach

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

An advanced, AI-powered grading platform designed to bring reliability, consistency, and transparency to automated academic assessment. This project moves beyond single-model limitations by implementing a sophisticated multi-agent architecture where a team of AI agents collaborates to grade student work. 

<div align="center">

*Placeholder for Project Demo GIF - A short, animated demonstration of the UI in action would go here.*

</div>

---

### 🤔 Why This Framework?

While single-model LLMs can grade assignments, they suffer from inherent non-determinism and can be easily influenced by subtle changes in prompt phrasing. This leads to inconsistent and unreliable results—a critical failure for educational tools. 

This framework solves that problem by simulating a human grading committee. By instantiating a team of AI agents with diverse personas (e.g., a "strict" grader, a "lenient" one), we introduce multiple perspectives. Their scores are then aggregated into a consensus, and their feedback is synthesized by a "meta-agent." This multi-agent approach significantly reduces bias and variance, leading to more robust, reliable, and trustworthy evaluations.

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
| 🚀 **MLOps & LMS Ready** | Designed with versioned feedback for CI/CD workflows, with planned integration for LMS like ILIAS and Canvas.
| 🖼️ **Multimodal Support** | Architected to support future evaluation of image-based answers, handwritten notes, and scientific sketches.

</div>

---

### 🛠️ Technology Stack

A detailed look at the technologies, frameworks, and libraries that power the AI Grading Framework.

<p align="center">
  <b>Core & Frontend</b><br>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
</p>

<p align="center">
  <b>AI & Machine Learning</b><br>
  <img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama" />
  <img src="https://img.shields.io/badge/FAISS-3B5998?style=for-the-badge&logo=facebook&logoColor=white" alt="FAISS" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/PyPDF2-D32F2F?style=for-the-badge" alt="PyPDF2" />
  <img src="https://img.shields.io/badge/Concurrent.futures-4B8BBE?style=for-the-badge" alt="concurrent.futures" />
</p>

<p align="center">
  <b>Database & Backend</b><br>
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white" alt="SQLAlchemy" />
  <img src="https://img.shields.io/badge/psycopg2-336791?style=for-the-badge" alt="psycopg2" />
</p>

<p align="center">
  <b>Infrastructure & DevOps</b><br>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white" alt="Git" />
  <img src="https://img.shields.io/badge/unittest-2C3E50?style=for-the-badge" alt="unittest" />
  <img src="https://img.shields.io/badge/PyYAML-CB172C?style=for-the-badge" alt="PyYAML" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow" />
</p>

---

## 🚀 Installation & Usage

**Prerequisites:** Ensure **Python 3.8+**, **Docker**, **PostgreSQL**, and **Ollama** are installed and running.

**1. Clone & Install**
```bash
# Clone the repository
git clone https://github.com/vedant-m/multi-agent-llm-grader.git
cd multi-agent-llm-grader

# Install Python dependencies
pip install -r requirements.txt
```

**2. Database & Model Setup**
```bash
# Connect to PostgreSQL and run these commands
CREATE DATABASE autograder_db;
CREATE USER vedant WITH PASSWORD 'vedant';
GRANT ALL PRIVILEGES ON DATABASE autograder_db TO vedant;

# Initialize the database schema
python init_db.py

# Pull the local LLM via Ollama
ollama pull mistral
```
> **Note:** Database credentials are set in `credentials.yaml`. Modify this file if you use different settings.

**3. Launch the Application**
```bash
# Ensure Docker and Ollama are running, then:
streamlit run app.py
```
Navigate to `http://localhost:8501` and start grading!

---

### 🔄 How It Works: The Grading Lifecycle

1.  **Upload:** The educator uploads a PDF containing the assignment questions, ideal answers, and a detailed grading rubric.
2.  **Submit:** Students (or the educator) upload their completed assignments.
3.  **Grade:** The educator initiates the grading process. The AI Grading Engine dispatches a team of agents to evaluate each submission against the rubric.
4.  **Review:** The initial AI-generated scores and feedback appear in the UI. The system flags grades where the agents had low consensus (high variance).
5.  **Correct (HITL):** The educator reviews the results, making any necessary corrections. Each correction is automatically saved and used to enrich the RAG system's knowledge base, making future grading even more accurate.

---

### 🗺️ Project Roadmap

- [ ] **Full MLOps Integration:** Implement an end-to-end MLflow pipeline for tracking experiments and versioning models/feedback.
- [ ] **LMS Integration:** Develop production-ready connectors for popular Learning Management Systems like ILIAS and Canvas.
- [ ] **Activate Multimodal Grading:** Implement the OCR and image-processing pipeline for grading graphical and handwritten submissions.
- [ ] **Analytics Dashboard:** Build a dashboard for educators to visualize class performance, question difficulty, and grading consistency.

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**. Please fork the repo and create a pull request, or open an issue with your suggestions.

---

## 👨‍💻 About the Author

This project was developed by **Vedant M.** as a dedicated effort to explore the frontiers of AI in education. Driven by a passion for building reliable and practical machine learning systems, this framework is a testament to the potential of multi-agent architectures.

### Connect with Me

Feel free to reach out, connect, or follow my work. I'm always open to discussions, collaborations, or a friendly chat about technology.

<p align="left">
  <a href="https://github.com/vedant-m" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
  </a>
  <a href="https://www.linkedin.com/in/your-linkedin-profile/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/>
  </a>
  <a href="mailto:your-email@example.com">
    <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Gmail"/>
  </a>
</p>

---

## 📜 License

Distributed under the **GNU General Public License v3.0**. See `LICENSE` for more information.
