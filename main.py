import os
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from vector import retriever

# Use a valid model name; adjust if you have access to others
llm = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "").strip('"').strip("'"),
    model="gpt-4o-mini",
    temperature=0,  # reduce hallucinations
)


template = """
You are an expert HR assistant. Answer ONLY using the information in the provided context. If the context is insufficient, say "I don't know" and suggest what information is needed.

Context (numbered chunks with sources):
{reviews}

Question: {question}

Instructions:
- Do not invent facts. Do not use outside knowledge.
- If numbers are present, copy them exactly from the context.
- Provide a concise answer and include bracket citations like [1], [2] that refer to the context chunks.
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    parts = []
    for i, d in enumerate(docs, start=1):
        src = getattr(d, "metadata", {}).get("source", "unknown")
        ridx = getattr(d, "metadata", {}).get("row_index")
        cidx = getattr(d, "metadata", {}).get("chunk_index")
        ident = f"row {ridx}" if ridx is not None else (f"chunk {cidx}" if cidx is not None else "")
        header = f"[{i}] Source: {src} {ident}".strip()
        parts.append(f"{header}\n{getattr(d, 'page_content', str(d))}")
    return "\n\n".join(parts)

# Bind retriever so it automatically fetches context from the vector DB
chain = (
    {
        "reviews": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
    }
    | prompt
    | llm
)

while True:
    print("\n\n--------------------------------------")
    question = input("Enter your HR question (q to quit): ")
    print ("\n\n")
    if question == "q":
        break

    result = chain.invoke({"question": question})
    print(result.content)