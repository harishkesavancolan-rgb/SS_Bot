"""
api/llm.py
----------
Sends retrieved chunks + user question to Mistral Mixtral 8x7B
and returns a grounded answer.

"Grounded" means Mistral only answers based on the chunks we provide.
"""

import os
import json
import boto3
from typing import List, Dict

# ── Config ────────────────────────────────────────────────────────────────────

# Best quality/cost Mistral model - uses signup credits
MODEL_ID = "mistral.mixtral-8x7b-v1:0"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
MAX_TOKENS = 1024

# ── System instructions (baked into prompt for Mistral) ───────────────────────

SYSTEM_INSTRUCTIONS = """You are a helpful assistant that ONLY answers questions using the provided document chunks.

IMPORTANT RULES:
- ONLY use information from the provided context chunks
- If the question cannot be answered from chunks, respond exactly: "I couldn't find that information in your uploaded documents."
- Do NOT use general knowledge, personal opinions, or external information
- Do NOT fabricate or hallucinate information
- Quote relevant chunks when possible
- Keep responses concise and factual"""

# ── Build Mistral prompt ─────────────────────────────────────────────────────

def _build_prompt(question: str, chunks: List[Dict]) -> str:
    """Builds Mistral chat template with context chunks."""
    if not chunks:
        return f"<s>[INST] {question} [/INST]"
    
    # Format chunks clearly
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        page = chunk.get("page_number", "?")
        doc = chunk.get("doc_id", "unknown")
        score = chunk.get("rerank_score", chunk.get("similarity_score", 0))
        text = chunk.get("text", "")
        context_parts.append(f"### Chunk {i} ({doc}, Page {page}, Score: {score:.2f})\n{text}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""<s>[INST] <<SYS>>
{SYSTEM_INSTRUCTIONS}

Context:
{context}
<</SYS>>

Question: {question} [/INST]"""
    
    return prompt

# ── Call Mistral Mixtral ──────────────────────────────────────────────────────

async def generate_answer(
    question: str,
    chunks: List[Dict],
    chat_history: List[Dict] = None,
) -> str:
    """Sends question + chunks to Mistral and returns grounded answer."""
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    
    # Build Mistral native prompt
    prompt = _build_prompt(question, chunks)
    
    # Mistral native request format
    body = json.dumps({
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.1,  # Low for grounded answers
        "top_p": 0.9
    })
    
    try:
        response = client.invoke_model(
            modelId=MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        
        response_body = json.loads(response["body"].read())
        answer = response_body["outputs"][0]["text"].strip()
        
    except Exception as e:
        print(f"[llm] ❌ Mistral error: {type(e).__name__}: {str(e)}")
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
