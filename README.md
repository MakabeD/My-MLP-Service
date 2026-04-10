# 🧠 Machine Learning Services (MLP Portfolio)

Welcome to the central repository for my Machine Learning services. These projects are built with robust version control (using Git and DVC) and MLOps best practices. The main goal of this repository is to deploy a set of model pipelines capable of solving real-life industry problems.

Currently, this repository hosts two main services:
1. **Credit Scoring Service**
2. **Mobile Churn Service**

While each service has its own specific execution steps and unique pipeline architecture, they all follow a standardized and scalable MLOps lifecycle.

## 🛠️ Tech Stack & Architecture

This project was built using a modern, end-to-end Machine Learning stack:
* **Core ML & Python:** PyTorch, Scikit-learn, Pydantic, Joblib, Pathlib.
* **MLOps & Tracking:** MLflow (experiment tracking), DVC (data version control), DAGsHub (remote storage), YAML (configuration management).
* **CI/CD & DevOps:** GitHub Actions (Workflows), Docker.
* **Cloud & Deployment:** Deployed entirely on Google Cloud Platform (Cloud Run & Artifact Registry).
* **API & Frontend:** FastAPI (Backend), Next.js, React, TypeScript, Tailwind CSS (Frontend).

## 🌐 Live Demos

You can interact with all the ML playgrounds and test the models directly on my portfolio site:
🔗 **[danielyepes-hub.vercel.app](https://danielyepes-hub.vercel.app)**

---

## 📚 Services Documentation Directory

Below you will find the specific documentation for each pipeline/service (including the API Gateway). Click on each link to see how to execute them, required YAML configurations, and model details:

* 🏦 **[Credit Scoring Service](python/credit-scoring/README.md)**
* 📱 **[Mobile Churn Service](python/mobile-churn/README.md)**