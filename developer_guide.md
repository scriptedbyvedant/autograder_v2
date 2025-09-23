
# Developer Guide: AI Grading Framework

## 1. Overview

This document provides a technical deep dive for developers contributing to the AI Grading Framework. It covers the system architecture, API specifications, core logic, deployment procedures, and testing guidelines.

---

## 2. System Architecture & Core Components

### 2.1. Technology Stack

The backend is built on a modern, scalable stack designed for intensive AI and machine learning workloads.

- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) for high-performance, asynchronous API endpoints.
- **AI Services**: [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai) for scalable model serving and ML operations.
- **AI Orchestration**: [LangChain](https://python.langchain.com/) for building and managing the multi-agent system.
- **LLM APIs**: [OpenAI](https://platform.openai.com/docs/api-reference) for accessing foundational models like GPT-4.
- **Deployment**: Docker, Google Cloud Run, and Firebase for containerization, serving, and infrastructure.

### 2.2. Core Logic Components

- **Grading Agent**: The primary agent responsible for interpreting the rubric and providing an initial grade and feedback. It is designed to be analytical and strictly follow the provided criteria.
- **Critic Agent**: A secondary agent that reviews the Grading Agent's output. It validates the reasoning, checks for hallucinations, and ensures the feedback aligns with the rubric. This adversarial setup is key to the system's reliability.
- **Rubric Validator**: A Pydantic model that enforces the structural integrity of grading rubrics before they are used in the grading process. This prevents errors from invalid or poorly formatted criteria.

---

## 3. API & Data Models

### 3.1. API Endpoints

The following are the core endpoints for the grading service.

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/grade` | `POST` | The primary endpoint that accepts a student submission and a rubric, and returns a grade.
| `/validate_rubric` | `POST` | A utility endpoint to check if a given rubric conforms to the required data model before use. |

### 3.2. Core Data Models

| Model | Fields | Description |
| :--- | :--- | :--- |
| `Submission` | `student_id: str`, `content: str` | Represents the work submitted by a student. |
| `Rubric` | `criteria: List[Dict]`, `scoring_scale: str` | Defines the criteria and scale for grading an assignment. |
| `GradeResponse`| `score: float`, `feedback: str`, `errors: List` | The structured output returned by the `/grade` endpoint. |

---

## 4. Deployment & Infrastructure

### 4.1. Docker

The entire application is containerized using Docker for consistent and reproducible deployments. The `Dockerfile` in the root directory contains the build steps.

### 4.2. Google Cloud

The application is designed to be deployed on **Google Cloud Run**, with container images stored in the **Google Artifact Registry**. Continuous deployment can be configured via Cloud Build triggers.

### 4.3. Firebase

**Firebase** is used for hosting supporting web assets and managing certain backend configurations and rules for the application.

---

## 5. Testing & Automation

### 5.1. Code Coverage with Android Test Orchestrator

When generating code coverage reports, use the following environment variables with the `gcloud` command to ensure the orchestrator saves the output to the correct path.

```bash
gcloud firebase test android run \
  --type instrumentation \
  --app your-app.apk \
  --test your-app-test.apk \
  --device model=TestDevice,version=AndroidVersion  \
  --environment-variables clearPackageData=true,coverage=true,coverageFilePath="/sdcard/Download/" \
  --directories-to-pull /sdcard/Download
```

### 5.2. UI Test Automation Rules

The following JSON configuration defines rules for the UI automation framework, allowing it to ignore certain elements during tests.

```json
[
  {
    "id": 1000,
    "contextDescriptor": {
      "condition": "element_present",
      "elementDescriptors": [
        {
          "resourceId": "my.app.package:id/ignored_screen"
        }
      ]
    },
    "actions": [
      {
        "eventType": "ALL_ELEMENTS_IGNORED"
      }
    ]
  },
  {
    "id": 1001,
    "contextDescriptor": {
      "condition": "element_present",
      "elementDescriptors": [
        {
          "resourceId": "my.app.package:id/main_screen"
        }
      ]
    },
    "actions": [
      {
        "eventType": "ELEMENT_IGNORED",
        "elementDescriptors": [
          {
            "resourceIdRegex": ".*:id/done"
          }
        ]
      }
    ]
  }
]
```
