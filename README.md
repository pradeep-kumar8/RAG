# 🔍 GenAI-LLM

### (Hugging Face, LangChain, Gemini, ChatGPT, FAISS, RAG)

Welcome to **GenAI-LLM**!  
This repository is a collection of powerful and modular AI applications built using state-of-the-art **Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)** techniques. Projects here leverage models and APIs like **Mistral 7B**, **Google Gemini 1.5**, **Meta LLaMA**, and others for solving real-world problems — from document QA to intelligent chatbots.

---

## 📚 Sub-Project Overview

---

### 1. **PDF Chatbot with Multi-LLM Support**
A powerful chatbot that allows you to **upload a PDF and ask questions** directly about its contents using the power of **LangChain + FAISS + Hugging Face** hosted models.

**Features:**
- Upload and parse PDFs.
- Choose from multiple Hugging Face-hosted LLMs (Mistral, DeepSeek, Qwen, LLaMA 3, Bloom).
- Embeds and retrieves data using FAISS.
- Semantic search + contextual answers using RetrievalQA chain.

📁 Project Path: [`/pdf-chatbot`](./pdf-chatbot)
---

## 🚧 Upcoming Projects

These projects are in progress and will soon be added to this repository:

### 🧠 Semantic Notes Search
> Upload your personal notes and search them semantically using embeddings + LLMs.

### 📜 Legal Doc Summarizer (RAG)
> Summarize and query lengthy legal documents using chunking + retrieval-augmented generation.

### 🏢 Internal Org Wiki Chatbot
> Upload your company's internal knowledge base and enable employees to chat with it privately.

### 🧾 Invoice Data Extractor
> Upload invoices and extract structured tabular data like vendor, date, amount, and more.

---

## ⚙️ Tech Stack

| Component      | Usage                                              |
|----------------|----------------------------------------------------|
| `LangChain`    | Prompt chaining, RAG pipelines, QA chains          |
| `Streamlit`    | UI for all applications                            |
| `FAISS`        | Embedding-based document retrieval                 |
| `HuggingFaceHub` | Access to LLMs like Mistral, LLaMA, DeepSeek     |
| `Google Gemini`| Text & Vision models for NLP and image reasoning   |
| `PyPDF`, `PIL` | PDF parsing and image handling                     |

---
