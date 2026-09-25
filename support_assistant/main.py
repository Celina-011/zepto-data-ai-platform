
import os
from typing import TypedDict

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from openai import OpenAI


DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = None

if not MOCK_LLM:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
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

def generate_llm_answer(prompt: str) -> str:
    if client is None:
        raise RuntimeError("OpenAI client is not configured.")

    last_error = None

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Zepto customer support assistant. "
                            "Answer only from the supplied policy context. "
                            "Never invent policies or use outside knowledge."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            answer = response.choices[0].message.content

            if not answer or not answer.strip():
                raise ValueError("The model returned an empty answer.")

            return answer.strip()

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"LLM failed after 3 attempts: {last_error}"
    )

def retrieve_and_answer(state: State):
    query = state["query"]

    # Step 1: Create query embedding
    q_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    # Step 2: Retrieve relevant policy chunks
    result = collection.query(
        query_embeddings=q_embedding,
        n_results=3
    )

    docs = result["documents"][0]
    metadatas = result["metadatas"][0]
    chunk_ids = result["ids"][0]

    if not docs:
        return {
            "answer": Answer(
                answer=(
                    "I could not find relevant information "
                    "in the available Zepto policies."
                ),
                sources=[],
                confidence=0.0
            )
        }

    # Step 3: Identify source documents
    sources = []

    for metadata, chunk_id in zip(metadatas, chunk_ids):
        if metadata and metadata.get("source"):
            source = metadata["source"]
        else:
            source = chunk_id.split("_chunk_")[0]

        if source not in sources:
            sources.append(source)

    # Step 4: Combine retrieved context
    context = "\n\n".join(docs)

    # Step 5: Structured prompt
    prompt = f"""
ROLE:
You are a helpful Zepto customer support assistant.

CONTEXT:
Use only the following retrieved Zepto policy information:

{context}

TASK:
Answer the user's question using the policy context.

If the answer is not available in the context,
clearly say that the available policies do not
provide the information.

Do not invent policies, prices, guarantees,
or exceptions. Do not use outside knowledge.

FEW-SHOT EXAMPLE:

User question:
Can I return an opened item?

Retrieved policy:
Opened items are non-returnable except in the
case of a manufacturing defect.

Expected answer:
Opened items are non-returnable except in the
case of a manufacturing defect.

FORMAT:
Return a clear, concise, customer-friendly answer.

LENGTH:
Use no more than 3 sentences.

USER QUESTION:
{query}
"""

    # Step 6: Generate answer
    if MOCK_LLM:
        # Deterministic mode; no external API call
        answer_text = (
            f"Based on the retrieved context: {docs[0]}"
        )
    else:
        # Real OpenAI mode
        answer_text = generate_llm_answer(prompt)

    # Step 7: Validate response using Pydantic
    validated_answer = Answer(
        answer=answer_text,
        sources=sources,
        confidence=0.75
    )

    return {
        "answer": validated_answer
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
