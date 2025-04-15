# 📄💬 PDF Chatbot with Multi-LLM Support

Interact with your **PDF documents** using the power of **large language models (LLMs)** like **Mistral**, **LLaMA 3**, **DeepSeek**, and more — all through a sleek **Streamlit** interface!

This project allows you to **upload a PDF**, select your favorite **Hugging Face-hosted LLM**, and then **ask questions** about the document's content in a conversational format.

---

## 🖼️ Screenshot

### UI
> ![image](https://github.com/user-attachments/assets/d4701129-4c42-41e7-b3b3-923d29403404)
### after upload PDF
> ![image](https://github.com/user-attachments/assets/b7559c43-8fde-49b8-8816-9a30f60ccb47)
### Getting answer from PDF
> ![image](https://github.com/user-attachments/assets/f3c4caea-e866-4954-8fb4-ca5f11330fae)

---

## ✨ Features

- 🗂️ Upload any **PDF** and get instant insights via chat
- 🤖 Choose from **multiple LLMs** hosted on **Hugging Face**
- 📄 Automatically splits & embeds text from uploaded PDFs
- 🧠 **FAISS** vector store for efficient document retrieval
- 🔐 Enter Hugging Face Token from sidebar (no hardcoding)
- 🧹 Clear conversation at any point to start over

---

## 🛠️ Tech Stack

| Component        | Description                                       |
|------------------|---------------------------------------------------|
| **Streamlit**     | UI and interaction layer                         |
| **LangChain**     | LLM orchestration and QA chain management        |
| **FAISS**         | Vector storage for document retrieval            |
| **HuggingFaceHub**| Connects to models like Mistral, LLaMA, Bloom    |
| **PyPDFLoader**   | Extracts and splits PDF into usable text chunks  |
| **Embeddings**    | HuggingFace sentence transformers                |

---

