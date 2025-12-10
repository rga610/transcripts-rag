"""Main Streamlit application for Financial Call Transcript RAG Assistant."""
import streamlit as st
from rag.document_processor import process_pdf_file
from rag.vector_store import store_document_chunks
from rag.qa_chain import answer_question
from db.conversations import create_conversation, add_message, get_conversation_messages
import uuid


# Page configuration
st.set_page_config(
    page_title="Financial Transcript RAG Assistant",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


def main():
    st.title("📊 Financial Call Transcript RAG Assistant")
    st.markdown("Upload earnings call transcripts (PDFs) and ask questions about management commentary, guidance, margins, and more.")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload Transcripts")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more earnings call transcript PDFs (max 10MB each)"
        )
        
        if uploaded_files:
            new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_files]
            
            if new_files:
                with st.spinner("Processing PDFs..."):
                    for file in new_files:
                        try:
                            # Process PDF
                            chunks = process_pdf_file(file)
                            
                            # Store in vector database
                            store_document_chunks(chunks)
                            
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
    if prompt := st.chat_input("Ask a question about the transcripts..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Create conversation if needed
        if st.session_state.conversation_id is None:
            st.session_state.conversation_id = create_conversation()
        
        # Save user message to database
        add_message(st.session_state.conversation_id, "user", prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = answer_question(prompt)
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

