import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory

# Import your core chain functions (these are based on your previous setup)
from rag_chain import (
    initialize_chain,
    chat_with_rag_and_tools,
    get_memory_summary,
    clear_memory,
)

# Streamlit page setup
st.set_page_config(page_title="LangChain Memory Chatbot", page_icon="🤖")
st.title("LangChain Chatbot with Memory")

# Initialize or load memory
@st.cache(allow_output_mutation=True)
def load_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

memory = load_memory()

# Initialize Chat Model
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    temperature=0.7
)

# User input
if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.text_input("Your message:", key="user_input")

# Buttons for control
col1, col2 = st.columns([1, 1])
if col1.button("Clear Chat"):
    memory.clear()
    st.session_state["history"] = []

if user_input:
    # Save user message to memory
    memory.save_context({"input": user_input}, {})
    
    # Build conversation history from memory
    memory_vars = memory.load_memory_variables({})
    history_messages = []
    if "history" in memory_vars and memory_vars["history"]:
        for msg in memory_vars["history"]:
            if hasattr(msg, "content") and hasattr(msg, "type"):
                if msg.type == "human":
                    history_messages.append(HumanMessage(content=msg.content))
                # You can extend here for AIMessage if needed
    
    # Call your chain with the message and history
    response = chat_with_rag_and_tools(
        user_message=user_input,
        memory=memory,
        chain=initialize_chain(),
        history=history_messages
    )
    
    # Save response to memory
    memory.save_context({"input": user_input}, {"output": response})
    
    # Add to session history for display
    st.session_state["history"].append(("User", user_input))
    st.session_state["history"].append(("Assistant", response))
    # Clear input box
    st.session_state["user_input"] = ""

# Display conversation history
if st.session_state["history"]:
    for speaker, message in st.session_state["history"]:
        if speaker == "User":
            st.markdown(f"**{speaker}:** {message}")
        else:
            st.markdown(f"**{speaker}:** {message}")

# Optional: Display memory summary
if st.button("Show Memory Summary"):
    summary = get_memory_summary(memory)
    st.write("**Memory Summary:**")
    st.write(summary)
