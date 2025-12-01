import streamlit as st
import os

# Import your chain functions from rag_chain
from rag_chain import (
    initialize_chain, 
    chat_with_rag_and_tools,
    get_memory_summary,
    clear_memory,
    get_memory_messages_list
)

# Page configuration
st.set_page_config(
    page_title="RAG Agent with Memory",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for conversation and chain
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain_initialized" not in st.session_state:
    st.session_state.chain_initialized = False

# Sidebar for configuration and controls
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Check API keys
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY") or not os.getenv("LANGCHAIN_API_KEY"):
        st.warning("⚠️ Please set OPENAI_API_KEY, PINECONE_API_KEY, and LANGCHAIN_API_KEY in secrets.")
    else:
        st.success("✅ API keys configured.")

    st.divider()

    # Memory management buttons
    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        clear_memory()
        st.session_state.chain_initialized = False
        st.success("✅ Chat history and memory cleared!")
        st.rerun()

    # Show current memory status
    with st.expander("📊 Memory Status"):
        mem_summary = get_memory_summary()
        history_preview = mem_summary.get("history", "")[:200] + "..." if mem_summary.get("history") else "No history yet."
        message_count = mem_summary.get("message_count", 0)
        st.markdown(f"**History Preview:**\n{history_preview}")
        st.markdown(f"**Messages in memory:** {message_count}")

    st.divider()
    # Demo prompts
    st.subheader("💡 Demo Prompts")
    st.caption(
        """
        **Test your setup:**
        - "What are my calorie targets given I weigh 75kg, have a desk job, and train 3 times per week?"
        - "Adjust my plan as I can only train 2 times per week now."
        - "What’s your advice based on my earlier info?"
        """
    )

# Main title
st.title("🤖 Body Logic - RAG Agent with Memory")
st.markdown(
    "Ask questions about your fitness goals. I use tools, retrieval-augmented generation, and conversational memory to provide personalized advice."
)

# Initialize chain once
if not st.session_state.chain_initialized:
    try:
        with st.spinner("🔧 Initializing RAG chain with LangChain memory..."):
            initialize_chain()
            st.session_state.chain_initialized = True
    except Exception as e:
        st.error(f"❌ Failed to initialize chain: {e}")
        st.stop()

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input prompt
if prompt := st.chat_input("Ask your question here..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            with st.spinner("🤖 Thinking..."):
                response = chat_with_rag_and_tools(prompt)
            placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_text = f"❌ Error: {e}"
            placeholder.error(error_text)
            st.session_state.messages.append({"role": "assistant", "content": error_text})

# Footer info
st.divider()
st.caption(
    "✨ Features: Retrieval from Pinecone • Tool calls (calorie estimator, calculator) • "
    "Conversation memory via LangChain's ConversationBufferMemory"
)
