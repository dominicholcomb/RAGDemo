import streamlit as st
import anthropic
import os
from pinecone import Pinecone
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI




# Check Streamlit Secrets first
ant_api = st.secrets["ANTHROPIC_API_KEY"] if "ANTHROPIC_API_KEY" in st.secrets else None
pc_api = st.secrets["PINECONE_API_KEY"] if "PINECONE_API_KEY" in st.secrets else None
oai_api = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else None

# Fallback to .env for local testing
if not all([ant_api, pc_api, oai_api]):
    load_dotenv()
    ant_api = os.getenv("ANTHROPIC_API_KEY")
    pc_api = os.getenv("PINECONE_API_KEY")
    oai_api = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=pc_api)
index_name = "domrag2"
index = pc.Index(index_name)

# Initialize the Claude API client
client_ant = anthropic.Anthropic(api_key=ant_api)


client_open = OpenAI(api_key=oai_api)

def get_embedding(text):
    """Generates an embedding using OpenAI"""
    response = client_open.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding


# Function to call Claude API
def query_claude(prompt, model="claude-3-7-sonnet-20250219", max_tokens=300):
    """Calls the Claude API with a given prompt and returns the response."""
    try:
        response = client_ant.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.6,
            system="You are Dominic Holcomb. Your role is to chat professionally and briefly with the user. "
                   "The user may be assessing your fit for a certain job opportunity. Speak to your experience when relevant, "
                   "in a way that paints the most compelling case that you are well suited for that particular role.\n\n"
                   "Notably, the person interested in your background may be interested in a specific role, so it's important "
                   "that before you express my experience broadly, you get more information on what the user may be looking for "
                   "so you can make sure the information you provide is the most relevant to them.\n\n"
                   "Keep answers very brief, as though it's a conversation for fun.",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text  # Extract Claude's response

    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return None

# Function to retrieve relevant chunks from Pinecone
def search_pinecone(query, top_k=5):
    """Searches Pinecone for the most relevant chunks."""
    query_embedding = get_embedding(query)  # Assume get_embedding() is defined elsewhere
    
    results = index.query(
        vector=query_embedding, 
        top_k=top_k, 
        include_metadata=True
    )
    
    return results.get("matches", [])

# Function to generate response using Pinecone & Claude
def generate_response_from_pinecone(query):
    """Searches Pinecone, formats context, and queries Claude."""
    retrieved_chunks = search_pinecone(query)

    if not retrieved_chunks:
        return "I couldn't find relevant information."

    # Format retrieved chunks into context
    context = "\n\n".join(
        f"Source: {match['metadata'].get('source', 'Unknown')}\nChunk: {match['metadata'].get('text', '[No text available]')}"
        for match in retrieved_chunks
    )

    # Construct the prompt for Claude
    prompt = f"""
    Use the following retrieved information to answer the question as if you are Dominic.

    Retrieved Information:
    {context}

    Question: {query}
    Answer:
    """

    # Call Claude API and return response
    return query_claude(prompt)

# ------------- STREAMLIT UI ---------------
st.set_page_config(page_title="Dominic RAG LLM", page_icon="🤖")

st.title("Dominic RAG LLM")
st.write("Enter a question below to chat with my Claude-RAG solution that pretends to be me!")

# Initialize chat history if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
query = st.chat_input("Ask me anything...")
if query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate AI response with RAG
    response = generate_response_from_pinecone(query)

    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
