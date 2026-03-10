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

# Claude 3 Haiku — fast and cheap, perfect for our use case
CLAUDE_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
AWS_REGION      = os.environ.get("AWS_REGION", "us-east-1")
MAX_TOKENS      = 1024   # max length of Claude's response


# ── System prompt ─────────────────────────────────────────────────────────────

# The system prompt tells Claude HOW to behave
# We instruct it to:
#   1. Only use the provided context
#   2. Be honest when it doesn't know
#   3. Be concise and clear
SYSTEM_PROMPT = """You are a helpful assistant that answers questions 
based strictly on the provided document chunks.

Rules:
- Only use information from the provided context chunks
- If the answer is not in the chunks, say "I couldn't find that in the provided document"
- Be concise and clear
- Do not make up information
- Reference which part of the document supports your answer"""


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

    # Build the current message with context
    current_prompt = _build_prompt(question, chunks)

    # Build messages list
    # If there's chat history, include it for context
    # This is what gives the chatbot memory within a session!
    messages = []

    if chat_history:
        # Add previous messages from this session
        # This lets Claude remember earlier parts of the conversation
        for msg in chat_history[-6:]:   # last 6 messages = 3 exchanges
            messages.append({
                "role"   : msg["role"],
                "content": msg["content"],
            })

    # Add the current question with retrieved context
    messages.append({
        "role"   : "user",
        "content": current_prompt,
    })

    # Call Claude Haiku via Bedrock
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens"       : MAX_TOKENS,
        "system"           : SYSTEM_PROMPT,
        "messages"         : messages,
    })

    response = client.invoke_model(
        modelId     = CLAUDE_MODEL_ID,
        body        = body,
        contentType = "application/json",
        accept      = "application/json",
    )

    response_body = json.loads(response["body"].read())
    answer        = response_body["content"][0]["text"]

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