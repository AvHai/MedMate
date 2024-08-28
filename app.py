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
    custom_prompt_template = """Use the following context to answer the user's question about medical conditions. If the answer cannot be found in the context, state that you don't know.

      Context: Information about various medical conditions, including their short descriptions, symptoms, precautions, treatments, and types of doctors to consult.

      Question: {question}

      Provide the most relevant information based on the context. If the answer is not available, respond with "I don't know."
      Helpful answer:
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
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    qa_prompt = set_custom_prompt()
    qa_chain = retrieval_qa_chain(custom_llm, qa_prompt, db)

    return qa_chain

# Streamlit code
def main():
    st.title("DiagnosAI Bot")

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "end_chat" not in st.session_state:
        st.session_state.end_chat = False

    # User input and button to get the answer
    user_query = st.text_input("What are your symptoms?", key="user_query")

    if st.button("Get Answer"):
        if user_query:
            with st.spinner("Processing..."):
                qa_chain = qa_bot()
                res = qa_chain({"query": user_query})
                answer = res["result"]
                sources = res["source_documents"]

                # Update chat history in session state
                st.session_state.chat_history.append(f"You: {user_query}")
                st.session_state.chat_history.append(f"Bot: {answer}")

                # Display chat history
                for message in st.session_state.chat_history:
                    st.write(message)

                # Optionally display sources:
                # if sources:
                #     st.write("*Sources:*")
                #     for doc in sources:
                #         st.write(doc)
                # else:
                #     st.write("*Sources:* No sources found")
        else:
            st.warning("Please enter a query.")

    # End Chat button
    if st.button("End Chat"):
        st.session_state.end_chat = True
        st.write("Chat ended.")

if __name__ == "__main__":
    main()
