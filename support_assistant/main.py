
import os
from typing import TypedDict

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END


DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=os.path.join(os.path.dirname(__file__), "chroma_db"))
collection = client.get_or_create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"},
)


def ingest():
    chunk_texts = []
    chunk_ids = []
    metadatas = []

    chunk_size = 500
    overlap = 100

    for filename in sorted(os.listdir(DOCS_DIR)):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(DOCS_DIR, filename)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        source_id = filename.replace(".txt", "")

        # Split the document into overlapping chunks
        start = 0
        chunk_number = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunk_texts.append(chunk)
                chunk_ids.append(
                    f"{source_id}_chunk_{chunk_number}"
                )
                metadatas.append({"source": source_id})
                chunk_number += 1

            start += chunk_size - overlap

    if not chunk_texts:
        print("No policy documents found.")
        return

    embeddings = model.encode(
        chunk_texts,
        normalize_embeddings=True
    ).tolist()

    collection.upsert(
        ids=chunk_ids,
        documents=chunk_texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Ingested {len(chunk_texts)} policy chunks.")
ingest()

class Answer(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class AskRequest(BaseModel):
    query: str


class State(TypedDict, total=False):
    query: str
    intent: str
    answer: Answer


KEYWORDS = [
    "delivery", "return", "refund", "membership", "tracking",
    "cancel", "gift card", "support hours"
]


def classify_intent(state: State):
    query = state["query"].lower()
    intent = "policy_question" if any(k in query for k in KEYWORDS) else "general_question"
    return {"intent": intent}

def retrieve_and_answer(state: State):
    query = state["query"]

    q_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    result = collection.query(
        query_embeddings=q_embedding,
        n_results=3
    )

    docs = result["documents"][0]
    metadatas = result["metadatas"][0]
    chunk_ids = result["ids"][0]

    # Get the original document names safely
    sources = []

    for metadata, chunk_id in zip(metadatas, chunk_ids):
        if metadata is not None and metadata.get("source"):
            source = metadata["source"]
        else:
            source = chunk_id.split("_chunk_")[0]

        if source not in sources:
            sources.append(source)

    top = docs[0]

    answer = f"Based on the retrieved context: {top}"

    return {
        "answer": Answer(
            answer=answer,
            sources=sources,
            confidence=1.0
        )
    }



def direct_answer(state: State):
    return {"answer": Answer(
        answer="I can only answer questions about Zepto policies right now.",
        sources=[],
        confidence=1.0,
    )}


def route(state: State):
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


graph_builder = StateGraph(State)
graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("retrieve_and_answer", retrieve_and_answer)
graph_builder.add_node("direct_answer", direct_answer)
graph_builder.set_entry_point("classify_intent")
graph_builder.add_conditional_edges(
    "classify_intent",
    route,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    },
)
graph_builder.add_edge("retrieve_and_answer", END)
graph_builder.add_edge("direct_answer", END)
graph = graph_builder.compile()


app = FastAPI(title="Zepto Policy Support Assistant")


@app.post("/ask", response_model=Answer)
def ask(request: AskRequest):
    result = graph.invoke({"query": request.query})
    return result["answer"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
