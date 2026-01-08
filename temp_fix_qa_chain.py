from pathlib import Path

path = Path("rag/qa_chain.py")
text = path.read_text()
marker = '"""Question-answering chain using LangChain and OpenAI."""'
first = text.find(marker)
second = text.find(marker, first + len(marker)) if first != -1 else -1
if second != -1:
    trimmed = text[:second]
    path.write_text(trimmed.rstrip() + "\n")
else:
    path.write_text(text.rstrip() + "\n")
