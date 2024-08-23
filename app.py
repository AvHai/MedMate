import streamlit as st
import requests
import time
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"
HF_ACCESS_TOKEN = "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"
headers = {"Authorization": f"Bearer {HF_ACCESS_TOKEN}"}

DB_FAISS_PATH = '/content/vectorstoredb'
SVC_MODEL_PATH = '/content/mlmodel.pkl'

# Function to handle API requests with retry mechanism
def query(payload):
    retries = 3
    backoff_factor = 2

    for i in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait_time = backoff_factor ** i
                st.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Max retries reached. Unable to process the request.")

# Custom LLM function
def custom_llm(input_text):
    payload = {
        "inputs": input_text,
        "parameters": {"max_new_tokens": 128, "temperature": 0.1}
    }
    response = query(payload)
    
    # Check if response is a list and extract the generated text
    if isinstance(response, list) and len(response) > 0:
        generated_text = response[0].get("generated_text", "")
    else:
        generated_text = "No response from model."
    
    return {"generated_text": generated_text}

def set_custom_prompt():
    custom_prompt_template = """Use the following pieces of information to answer the user's question on symptoms they have.
    If you don't know the answer don't try to make up an answer.
    Briefly describe the disease and treatments of the disease user has and appropriate type of doctor.
    
    Context: {context}
    Question: {question}

    Only return the helpful answer below.
    Helpful answer:
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm_func, prompt, db):
    def custom_chain(query):
        retriever = db.as_retriever(search_kwargs={'k': 2})
        documents = retriever.get_relevant_documents(query['query'])
        context = " ".join([doc.page_content for doc in documents])
        formatted_prompt = prompt.format(context=context, question=query['query'])
        
        llm_response = llm_func(formatted_prompt)
        return {"result": llm_response.get("generated_text", "No response"), "source_documents": documents}

    return custom_chain

# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
    qa_prompt = set_custom_prompt()
    qa_chain = retrieval_qa_chain(custom_llm, qa_prompt, db)

    return qa_chain

# Streamlit code
def main():
    st.title("DiagnosAI Bot")

    user_query = st.text_input("Hi, Welcome to DiagnosAI Bot. What are your symptoms?")

    if st.button("Get Answer"):
        if user_query:
            with st.spinner("Processing..."):
                qa_chain = qa_bot()
                res = qa_chain({"query": user_query})
                answer = res["result"]
                sources = res["source_documents"]

                st.write("*Answer:*", answer)
                if sources:
                    st.write("*Sources:*", sources)
                else:
                    st.write("*Sources:* No sources found")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
