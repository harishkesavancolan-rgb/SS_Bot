"""
api/llm.py
----------
Sends retrieved chunks + user question to Claude Haiku
and returns a grounded answer.

"Grounded" means Claude only answers based on the
chunks we provide — not from its training data.
This ensures answers are always traceable back to
the user's actual PDF.
"""

import os
import json
import boto3
from typing import List, Dict


# ── Config ────────────────────────────────────────────────────────────────────

# Amazon Nova Lite — fast, free, no approval needed
CLAUDE_MODEL_ID = "us.amazon.nova-lite-v1:0"
AWS_REGION      = os.environ.get("AWS_REGION", "us-east-1")
MAX_TOKENS      = 1024   # max length of Claude's response


# ── System prompt ─────────────────────────────────────────────────────────────

# The system prompt tells Claude HOW to behave
# We instruct it to:
#   1. Only use the provided context
#   2. Be honest when it doesn't know
#   3. Be concise and clear
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided document chunks.

Rules:
- Answer questions directly and naturally
- For greetings or casual messages, respond briefly and friendly, then mention what documents are available
- Only use information from the provided context chunks for factual questions
- If the answer is not in the chunks, say "I couldn't find that in the provided documents"
- Give detailed answers for specific questions
- Do not make up information"""


# ── Build the prompt ──────────────────────────────────────────────────────────

def _build_prompt(question: str, chunks: List[Dict]) -> str:
    """
    Builds the full prompt we send to Claude.

    Format:
        Context:
        [Chunk 1 - Page 3]
        text of chunk 1...

        [Chunk 2 - Page 7]
        text of chunk 2...

        Question: What does Sun Tzu say about deception?
    
    Why format it this way?
    Claude performs better when context is clearly
    separated from the question.
    """
    # Build context section from chunks
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        page    = chunk.get("page_number", "?")
        doc     = chunk.get("doc_id", "unknown")
        score   = chunk.get("rerank_score", chunk.get("similarity_score", 0))
        text    = chunk.get("text", "")

        context_parts.append(
            f"[Chunk {i} — {doc} | Page {page} | Score {score}]\n{text}"
        )

    context = "\n\n".join(context_parts)

    return f"""Context:
{context}

Question: {question}

Answer based only on the context above:"""


# ── Call Claude Haiku ─────────────────────────────────────────────────────────

async def generate_answer(
    question      : str,
    chunks        : List[Dict],
    chat_history  : List[Dict] = None,
) -> str:
    """
    Sends the question + chunks to Claude Haiku and returns the answer.

    Parameters
    ----------
    question     : the user's question
    chunks       : reranked chunks from retriever.py
    chat_history : previous messages in this session
                   format: [{"role": "user", "content": "..."},
                            {"role": "assistant", "content": "..."}]

    Returns
    -------
    Claude's answer as a string
    """
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    current_prompt = _build_prompt(question, chunks) if chunks else question

    # Build messages list including chat history
    messages = []
    if chat_history:
        for msg in chat_history[-6:]:
            messages.append({
                "role"   : msg["role"],
                "content": [{"text": msg["content"]}],
            })

    messages.append({
        "role"   : "user",
        "content": [{"text": current_prompt}],
    })

    # Nova request format (official AWS docs format)
    body = json.dumps({
        "schemaVersion"   : "messages-v1",
        "messages"        : messages,
        "system"          : [{"text": SYSTEM_PROMPT}],
        "inferenceConfig" : {
            "maxTokens"  : MAX_TOKENS,
            "temperature": 0.7,
            "topP"       : 0.9,
        }
    })

    try:
        response = client.invoke_model(
            modelId     = CLAUDE_MODEL_ID,
            body        = body,
            contentType = "application/json",
            accept      = "application/json",
        )
    except Exception as e:
        print(f"[llm] ❌ invoke_model error: {type(e).__name__}: {str(e)}")
        raise

    response_body = json.loads(response["body"].read())
    answer        = response_body["output"]["message"]["content"][0]["text"]

    print(f"[llm] generated answer ({len(answer)} chars)")
    return answer


# ── Format final response ─────────────────────────────────────────────────────

def build_response(answer: str, sources: List[Dict]) -> Dict:
    """
    Builds the final response object returned to the user.

    Format:
    {
        "answer": "Sun Tzu believed deception was...",
        "sources": [
            {
                "chunk_id"   : "ArtOfWar::chunk_0042",
                "display"    : "ArtOfWar.pdf | Page 3 | Score: 0.89",
                "text"       : "All warfare is based on deception...",
                "pdf_title"  : "ArtOfWar.pdf",
                "page_number": 3,
                "score"      : 0.89
            }
        ]
    }

    The frontend uses:
        display  → text shown on the hyperlink
        text     → exact chunk content shown in popup when clicked
        chunk_id → unique identifier for this chunk
    """
    return {
        "answer" : answer,
        "sources": [
            {
                "chunk_id"   : source["chunk_id"],
                "display"    : source["display"],
                "text"       : source["text"],       # ← exact chunk text
                "pdf_title"  : source["pdf_title"],
                "page_number": source["page_number"],
                "score"      : source["score"],
            }
            for source in sources
        ],
    }