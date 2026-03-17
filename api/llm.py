"""
api/llm.py
----------
Sends retrieved chunks + user question to Meta Llama 3.1 8B Instruct
and returns a grounded answer.

"Grounded" means Llama only answers based on the chunks we provide.
"""

import os
import json
import boto3
from typing import List, Dict

# ── Config ────────────────────────────────────────────────────────────────────

# Best quality/cost Llama model - uses signup credits
MODEL_ID = "meta.llama3-1-8b-instruct-v1:0"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
MAX_TOKENS = 1024

# ── System instructions ───────────────────────────────────────────────────────

SYSTEM_INSTRUCTIONS = """You are a helpful assistant that ONLY answers questions using the provided document chunks.

IMPORTANT RULES:
- ONLY use information from the provided context chunks
- If the question cannot be answered from chunks, respond exactly: "I couldn't find that information in your uploaded documents."
- Do NOT use general knowledge, personal opinions, or external information
- Do NOT fabricate or hallucinate information
- Quote relevant chunks when possible
- Keep responses concise and factual"""

# ── Build Llama 3 prompt ──────────────────────────────────────────────────────

def _build_prompt(question: str, chunks: List[Dict]) -> str:
    """Builds Llama 3 chat template with context chunks."""
    # Format chunks clearly
    if chunks:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            page = chunk.get("page_number", "?")
            doc = chunk.get("doc_id", "unknown")
            score = chunk.get("rerank_score", chunk.get("similarity_score", 0))
            text = chunk.get("text", "")
            context_parts.append(f"### Chunk {i} ({doc}, Page {page}, Score: {score:.2f})\n{text}")

        context = "\n\n".join(context_parts)
        user_content = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        user_content = question

    # Llama 3 chat template
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_INSTRUCTIONS}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    return prompt

# ── Call Llama 3.1 8B ─────────────────────────────────────────────────────────

async def generate_answer(
    question: str,
    chunks: List[Dict],
    chat_history: List[Dict] = None,
) -> str:
    """Sends question + chunks to Llama 3.1 8B and returns grounded answer."""
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    prompt = _build_prompt(question, chunks)

    # Llama 3 native request format
    body = json.dumps({
        "prompt": prompt,
        "max_gen_len": MAX_TOKENS,
        "temperature": 0.1,  # Low for grounded answers
        "top_p": 0.9,
    })

    try:
        response = client.invoke_model(
            modelId=MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )

        response_body = json.loads(response["body"].read().decode("utf-8"))
        answer = response_body["generation"].strip()

    except Exception as e:
        print(f"[llm] ❌ Llama error: {type(e).__name__}: {str(e)}")
        answer = "I couldn't process your request right now. Please try again."

    print(f"[llm] generated answer ({len(answer)} chars)")
    return answer

# ── Format final response ─────────────────────────────────────────────────────

def build_response(answer: str, sources: List[Dict]) -> Dict:
    """Builds final response with answer + source citations."""
    return {
        "answer": answer,
        "sources": [
            {
                "chunk_id": source["chunk_id"],
                "display": source["display"],
                "text": source["text"],
                "pdf_title": source["pdf_title"],
                "page_number": source["page_number"],
                "score": source["score"],
            }
            for source in sources
        ],
    }