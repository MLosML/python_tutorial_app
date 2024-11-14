!pip install -qqq -U streamlit
!npm install -qqq -U localtunnel
pip install streamlit langchain langchain-community faiss-cpu sentence-transformers langchain-groq requests beautifulsoup4

import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup

# Function to download and parse the Python tutorial content
@st.cache_resource
def load_tutorial_documents():
    url = "https://docs.python.org/3/tutorial/index.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Collect tutorial section links
    tutorial_links = [a['href'] for a in soup.select("a[href]") if "tutorial" in a['href']]

    # Download each section of the tutorial for processing
    tutorial_texts = []
    for link in tutorial_links[:10]:  # Limiting to the first 10 pages for testing
        page_url = f"https://docs.python.org/3/tutorial/{link}"
        page_response = requests.get(page_url)
        page_soup = BeautifulSoup(page_response.text, 'html.parser')
        tutorial_texts.append(page_soup.get_text())

    # Convert the collected texts into Document objects
    documents = [Document(page_content=text) for text in tutorial_texts]
    return documents

# Load documents
documents = load_tutorial_documents()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
split_docs = text_splitter.split_documents(documents)

# Initialize the embedding model and create a FAISS vector store
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# Create the FAISS vector store
vector_db = FAISS.from_documents(split_docs, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# Initialize the language model
llm = ChatGroq(
    temperature=0,
    groq_api_key="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",  # Ensure you have this in your Streamlit secrets
    model_name="mixtral-8x7b-32768"
)

# Set up memory for conversation
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

# Define the prompt template
template = """You are a helpful chatbot having a conversation with a human. Answer the question based only on the following context and previous conversation. Keep your answers short and succinct.

Previous conversation:
{chat_history}

Context to answer question:
{context}

New human question: {question}
Response:"""
prompt = PromptTemplate(template=template, input_variables=["context", "question", "chat_history"])

# Set up the conversational retrieval chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=False,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# Streamlit app interface
st.title("Python Tutorial Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me anything about Python!"):

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Thinking..."):

        # Send question to chain to get answer
        answer = chain.invoke({"question": prompt})

        # Extract answer from dictionary returned by chain
        response = answer["answer"]

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        import requests

# Define the actual URL for the API request
url = "https://api.example.com/data"  # Replace with your actual API endpoint

# Set up the headers with the API key
headers = {"Authorization": f"Bearer {'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'}"}

# Make the request
try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raises an error for bad responses (4xx or 5xx status codes)

    # Process the response
    data = response.json()  # Assuming the API returns JSON
    print("Response data:", data)
except requests.exceptions.RequestException as e:
    print("An error occurred:", e)
!streamlit run rag_app.py &>/content/logs.txt & npx localtunnel --port 8501 & curl ipv4.icanhazip.com

