"""Main Streamlit application for the SOP RAG Assistant."""
import streamlit as st
from rag.document_processor import process_pdf_file
from rag.vector_store import store_document_chunks
from rag.qa_chain import answer_question
from db.conversations import create_conversation, add_message, get_conversation_messages
import uuid


# Page configuration
st.set_page_config(
    page_title="SOP RAG Assistant",
    page_icon="🧾",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


def start_new_conversation():
    """Start a new conversation and reset session state."""
    st.session_state.conversation_id = create_conversation()
    st.session_state.messages = []
    st.session_state.uploaded_files = []
    st.rerun()


def main():
    st.title("🧾 SOP RAG Assistant")
    st.markdown("Upload SOP documents (PDFs) and ask structured, procedure-focused questions to get precise and accountable answers.")
    
    # Create conversation if it doesn't exist
    if st.session_state.conversation_id is None:
        st.session_state.conversation_id = create_conversation()
    
    # Sidebar for file upload and conversation management
    with st.sidebar:
        # New conversation button
        if st.button("🆕 New Conversation", use_container_width=True, type="primary"):
            start_new_conversation()
        
        st.markdown("---")
        st.header("📁 Upload SOPs")
        
        # Show current conversation info
        if st.session_state.conversation_id:
            st.caption(f"Conversation ID: {str(st.session_state.conversation_id)[:8]}...")
        
        uploaded_files = st.file_uploader(
            "Choose SOP PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more SOP PDFs (max 10MB each). Files are linked to this conversation."
        )
        
        if uploaded_files:
            new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_files]
            
            if new_files:
                # Ensure we have a conversation_id
                if st.session_state.conversation_id is None:
                    st.session_state.conversation_id = create_conversation()
                
                with st.spinner("Processing SOP PDFs..."):
                    for file in new_files:
                        try:
                            # Process PDF
                            chunks = process_pdf_file(file)
                            
                            # Store in vector database (linked to conversation)
                            store_document_chunks(chunks, st.session_state.conversation_id)
                            
                            st.session_state.uploaded_files.append(file.name)
                            st.success(f"✅ Processed: {file.name} ({len(chunks)} chunks)")
                        except Exception as e:
                            st.error(f"❌ Error processing {file.name}: {str(e)}")
        
        if st.session_state.uploaded_files:
            st.markdown("---")
            st.subheader("Uploaded Files")
            for filename in st.session_state.uploaded_files:
                st.text(f"• {filename}")
    
    # Main chat interface
    st.markdown("---")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about these SOPs..."):
        # Ensure we have a conversation_id
        if st.session_state.conversation_id is None:
            st.session_state.conversation_id = create_conversation()
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Save user message to database
        add_message(st.session_state.conversation_id, "user", prompt)
        
        # Generate response (scoped to this conversation)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = answer_question(prompt, st.session_state.conversation_id)
                    st.markdown(response)
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Save assistant message to database
                    add_message(st.session_state.conversation_id, "assistant", response)
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()

