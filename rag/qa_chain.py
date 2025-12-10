"""Question-answering chain using LangChain and OpenAI."""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import OPENAI_API_KEY, OPENAI_MODEL
from rag.embeddings import get_embedding_model
from rag.vector_store import search_similar_chunks
from typing import List, Dict


def get_llm():
    """Get the OpenAI LLM model."""
    return ChatOpenAI(
        model=OPENAI_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0
    )


def answer_question(question: str, conversation_history: List[Dict] = None) -> str:
    """Answer a question using RAG pipeline.
    
    Args:
        question: User's question
        conversation_history: Previous messages in the conversation
    
    Returns:
        Answer string
    """
    # Generate query embedding
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.embed_query(question)
    
    # Retrieve relevant chunks
    retrieved_chunks = search_similar_chunks(query_embedding)
    
    if not retrieved_chunks:
        return "I couldn't find any relevant information in the uploaded transcripts. Please make sure you've uploaded PDF files first."
    
    # Build context from retrieved chunks
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(f"[From {chunk['filename']}]\n{chunk['chunk_text']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Build prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a financial analyst assistant helping users understand earnings call transcripts.
        
Your task is to answer questions based ONLY on the provided context from the transcripts. 

CRITICAL RULES:
- Be precise and accurate - do not attribute numbers or facts to the wrong products, segments, or time periods
- If asked about a specific product/segment, only use numbers explicitly stated for that product/segment
- Distinguish between what is explicitly stated vs what you might infer
- If the context mentions a broader category (e.g., "Data Center revenue") but not the specific item asked about, clarify this distinction
- If the context doesn't contain the exact information requested, say so clearly rather than inferring
- Be concise and analytical
- Always cite which transcript the information came from (use the filename)
- Do not make up information or use knowledge outside the provided context"""),
        ("human", """Context from transcripts:
{context}

Question: {question}

Please provide a clear, precise answer based ONLY on what is explicitly stated in the context above. If the context doesn't contain the exact information requested, say so rather than making assumptions.""")
    ])
    
    # Format prompt
    messages = prompt_template.format_messages(context=context, question=question)
    
    # Get LLM response
    llm = get_llm()
    response = llm.invoke(messages)
    
    return response.content

