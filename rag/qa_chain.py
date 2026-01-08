"""Question-answering chain using LangChain and OpenAI."""
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL
from rag.embeddings import get_embedding_model
from rag.vector_store import search_similar_chunks


def get_llm():
    """Get the OpenAI LLM model."""
    return ChatOpenAI(
        model=OPENAI_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
    )


def _format_citation(metadata: Dict[str, str]) -> str:
    filename = metadata.get("source_filename", "Unknown SOP")
    section = metadata.get("section_path") or metadata.get("section_title") or "Section"
    return f"{filename} § {section}"


def _format_context_parts(chunks: List[Dict]) -> str:
    parts = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        citation = _format_citation(metadata)
        sop_meta = metadata.get("sop_metadata") or {}
        meta_trail = []
        for key in ("sop_id", "version", "department", "last_updated"):
            if sop_meta.get(key):
                meta_trail.append(f"{key.replace('_', ' ').title()}: {sop_meta[key]}")

        prefix = f"[{citation}]"
        if meta_trail:
            prefix += " (" + "; ".join(meta_trail) + ")"

        chunk_text = chunk.get("chunk_text", "").strip()
        if chunk_text:
            parts.append(f"{prefix}\n{chunk_text}")

    return "\n\n---\n\n".join(parts)


def answer_question(question: str, conversation_id: str, conversation_history: List[Dict] = None) -> str:
    """Answer a question using the SOP-focused RAG pipeline."""
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.embed_query(question)

    retrieved_chunks = search_similar_chunks(query_embedding, conversation_id)
    if not retrieved_chunks:
        return "I could not find any relevant information in the uploaded SOPs. Please upload PDFs first."

    context = _format_context_parts(retrieved_chunks)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an SOP specialist assistant. Use only the provided SOP context to answer questions.
- Be precise and honest; do not fabricate details that are not in the context.
- Always include a Citations line that references the source filename and section path or title for the statements you use.
- If the context does not contain enough information, explain that you are unable to answer rather than guessing.""",
            ),
            (
                "human",
                """Context from SOPs:
{context}

Question: {question}

Answer clearly, cite the SOP section(s) you rely on, and mention if the answer is not available from the context.""",
            ),
        ]
    )

    messages = prompt_template.format_messages(context=context, question=question)

    llm = get_llm()
    response = llm.invoke(messages)
    return response.content
