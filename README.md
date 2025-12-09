# MedMate → Unified Medical-Assistant Bot


MedMate + EM_Bot — An AI-powered medical assistant combining symptom-based disease prediction, doctor-specialist recommendation, and interactive UI via Gradio, to help users get quick medical insights in an easy-to-use web-app format.

## 🧠 What it does / Motivation

This project unifies two efforts:

MedMate — a medical bot that predicts possible diseases based on user-entered symptoms. 
GitHub

EM_Bot — an AI chatbot that analyses symptoms, suggests likely diseases, and recommends which type of doctor a user should consult. 
GitHub

By combining them and providing a simple UI with Gradio, the goal is to offer a lightweight, accessible tool for users to describe symptoms in plain language and receive:

potential disease predictions

suggestions for relevant medical specialists / doctor types

foundational information on diseases (as per integrated medical data)

This addresses a common pain point: people often don’t know which kind of doctor to consult, or what to expect given their symptoms — especially before visiting a clinic.

## 🧰 Tech Stack & Architecture

- Frontend / UI: Gradio — chosen because it enables building quick interactive web interfaces around Python code / ML models without needing full-scale frontend development. 
gradio.app
+1

- Backend / Logic: Python — data processing, symptom analysis, model inference, disease-prediction logic (from MedMate and EM_Bot).

- Data / Storage: Medical data files / vector-store / lookup database (as in repositories) to support symptom-to-disease / doctor-suggestion mapping. 
GitHub
+1

- Machine Learning / NLP / Retrieval-Augmented Generation (RAG): Use of ML models (or simpler rule-based / vector-search) to associate symptom inputs with possible diseases & doctor types. This draws from EM_Bot’s design approach. 
GitHub
+1

Using Gradio makes the app immediately demo-able, lowering barrier for recruiters / non-technical users to test and assess the tool — which shows that you care about usability, not just back-end logic.

## ✅ What I Did ?

Built a functional demo UI with Gradio, enabling users to type symptoms and receive predictions and doctor-type suggestions.

Implemented backend logic to parse symptoms, run disease prediction / recommendation, and serve results via web interface.

Structured the project to separate data / logic / UI, which shows good software-engineering practices.

Demonstrated full-stack capability — from data/model logic to frontend UI, leading to a working prototype.

## 📂 Repository Structure 
``` text
/             ← Root of unified project  
  ├─ data/               ← Medical data, disease-symptom mapping, doctor-specialist mapping  
  ├─ vectorstore/        ← Preprocessed data / embeddings (if using vector-search / RAG)  
  ├─ backend/            ← Python modules for symptom analysis, disease prediction, doctor recommendation  
  ├─ app.py               ← Main entry point — launches Gradio UI & integrates backend logic  
  ├─ requirements.txt     ← Python dependencies (including gradio, ML libs, etc.)  
  ├─ README.md            ← <- This file: project overview, how to run, usage, etc.  
  └─ LICENSE              ← e.g. MIT license  
```

## 🚀 How to Run / Demo
```
# Clone the repo  
git clone https://github.com/YourUsername/MedMate-EM_Bot.git  
cd MedMate-EM_Bot  

# (Optional) create & activate a venv  
python -m venv venv  
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies  
pip install -r requirements.txt  

# Run the application  
python app.py  
```

After running, open the displayed Gradio URL (usually http://localhost:7860
) in your browser to access the UI.

Using Gradio lets anyone test the tool easily — no web-dev setup required. 
gradio.app
+1

## 📝 Usage (What user does)

Enter symptoms (free-text) in the input box.

Bot processes input, runs analysis.

Returns predicted possible diseases.

Suggests which medical specialist / doctor type to consult.

Optionally provides basic disease information or advice / pointers (based on integrated data).
