"""
api/chat.py
-----------
FastAPI application — the main entry point for the chatbot.

Endpoints:
    POST /upload          → upload PDF to S3
    POST /chat            → ask a question
    GET  /sessions        → list user's sessions
    POST /sessions/new    → start a new session
    GET  /sessions/{id}   → get session history

Why FastAPI?
    - Built-in asyncio support (handles multiple users) ✅
    - Auto-generates API documentation at /docs       ✅
    - Works with AWS Lambda via Mangum wrapper        ✅
"""

import os
import uuid
import json
import boto3
import psycopg2
from psycopg2.extras   import RealDictCursor
from datetime          import datetime, timezone
from typing            import List, Optional

from fastapi            import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic           import BaseModel
from mangum             import Mangum

from api.retriever      import retrieve
from api.llm            import generate_answer, build_response


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "RAG Chatbot API",
    description = "Upload PDFs and chat with them",
    version     = "1.0.0",
)

# CORS — allows the frontend (browser) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = False,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

handler = Mangum(app, lifespan="off")

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET  = os.environ.get("S3_BUCKET",  "rag-pdf-uploads-harish")


# ── Request/Response models ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question   : str
    user_id    : str
    session_id : str

class ChatResponse(BaseModel):
    answer     : str
    sources    : List[dict]
    session_id : str

class NewSessionRequest(BaseModel):
    user_id : str

class NewSessionResponse(BaseModel):
    session_id : str
    created_at : str


# ── Database connection ───────────────────────────────────────────────────────

def _get_connection():
    return psycopg2.connect(
        host     = os.environ.get("DB_HOST"),
        dbname   = os.environ.get("DB_NAME",    "ragdb"),
        user     = os.environ.get("DB_USER",    "postgres"),
        password = os.environ.get("DB_PASSWORD"),
        port     = int(os.environ.get("DB_PORT", "5432")),
        sslmode  = "require",
    )


# ── Session table setup ───────────────────────────────────────────────────────

def ensure_sessions_table() -> None:
    """
    Creates the sessions table if it doesn't exist.

    Each session stores:
        session_id : unique ID for this conversation
        user_id    : which user owns this session
        messages   : full chat history as JSON
        created_at : when the session started
        updated_at : when the last message was sent
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id  TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    messages    JSONB DEFAULT '[]',
                    created_at  TIMESTAMP DEFAULT NOW(),
                    updated_at  TIMESTAMP DEFAULT NOW()
                );
            """)
            conn.commit()
    finally:
        conn.close()


# ── Session helpers ───────────────────────────────────────────────────────────

def get_session_history(session_id: str) -> List[dict]:
    """Fetches chat history for a session."""
    conn = _get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT messages FROM sessions WHERE session_id = %s",
                (session_id,)
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Session not found")
            return row["messages"] or []
    finally:
        conn.close()


def save_message(session_id: str, role: str, content: str) -> None:
    """
    Appends a message to the session history.

    role    : "user" or "assistant"
    content : the message text
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE sessions
                SET
                    messages   = messages || %s::jsonb,
                    updated_at = NOW()
                WHERE session_id = %s
            """, (
                json.dumps([{"role": role, "content": content}]),
                session_id,
            ))
            conn.commit()
    finally:
        conn.close()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Health check — confirms API is running."""
    return {"status": "RAG Chatbot API is running 🚀"}


@app.post("/sessions/new", response_model=NewSessionResponse)
async def new_session(request: NewSessionRequest):
    """
    Creates a new chat session for a user.

    Each session has its own independent chat history
    and its own isolated document space. PDFs uploaded
    in one session are never visible in another.

    Returns a session_id the frontend uses for all
    subsequent messages and uploads in this conversation.
    """
    ensure_sessions_table()

    session_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sessions (session_id, user_id, messages)
                VALUES (%s, %s, '[]')
            """, (session_id, request.user_id))
            conn.commit()
    finally:
        conn.close()

    print(f"[chat] new session created: {session_id}")
    return NewSessionResponse(session_id=session_id, created_at=created_at)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint — the heart of the chatbot.

    Flow:
        1. Load session history
        2. Retrieve relevant chunks scoped to this session only
        3. Generate answer (Amazon Nova Lite)
        4. Save messages to session
        5. Return answer + source hyperlinks

    Retrieval is scoped to request.session_id so each chat
    session only searches documents uploaded in that session.
    """
    # 1. Load session history for context
    history = get_session_history(request.session_id)

    # 2. Retrieve relevant chunks — scoped to this session
    retrieval = await retrieve(
        question   = request.question,
        user_id    = request.user_id,
        session_id = request.session_id,   # ← session-scoped retrieval
    )

    if not retrieval["chunks"]:
        raise HTTPException(
            status_code = 404,
            detail      = "No relevant content found in your documents"
        )

    # 3. Generate answer with Amazon Nova Lite
    answer = await generate_answer(
        question     = request.question,
        chunks       = retrieval["chunks"],
        chat_history = history,
    )

    # 4. Save both messages to session history
    save_message(request.session_id, "user",      request.question)
    save_message(request.session_id, "assistant", answer)

    # 5. Return answer + sources
    response = build_response(answer, retrieval["sources"])

    return ChatResponse(
        answer     = response["answer"],
        sources    = response["sources"],
        session_id = request.session_id,
    )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """
    Returns the full chat history for a session.

    The frontend uses this to restore a conversation
    when the user comes back to an existing session.
    """
    history = get_session_history(session_id)
    return {"session_id": session_id, "messages": history}


@app.get("/sessions")
async def list_sessions(user_id: str):
    """
    Lists all sessions for a user.

    The frontend shows these as a list of past conversations
    the user can click to resume.
    """
    conn = _get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    session_id,
                    created_at,
                    updated_at,
                    jsonb_array_length(messages) AS message_count
                FROM sessions
                WHERE user_id = %s
                ORDER BY updated_at DESC
            """, (user_id,))
            sessions = [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

    return {"sessions": sessions}


@app.post("/upload")
async def upload_pdf(
    user_id    : str,
    session_id : str,              # ← NEW: ties upload to a specific session
    file       : UploadFile = File(...),
):
    """
    Uploads a PDF to S3, scoped to a specific session.

    S3 key format: user_id/session_id/filename.pdf

    This means:
      - Each session has its own isolated document space
      - Chat session A can never retrieve docs from session B
      - Lambda reads the session_id from the S3 key and stores
        it in chunk metadata for filtered retrieval

    S3 automatically triggers Lambda which:
        chunks → embeds → stores in pgvector with session_id
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code = 400,
            detail      = "Only PDF files are allowed"
        )

    s3_client = boto3.client("s3", region_name=AWS_REGION)
    s3_key    = f"{user_id}/{session_id}/{file.filename}"   # ← session-scoped path

    try:
        s3_client.upload_fileobj(file.file, S3_BUCKET, s3_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    print(f"[chat] uploaded {file.filename} for user {user_id} / session {session_id}")
    return {
        "message"   : f"'{file.filename}' uploaded successfully",
        "s3_key"    : s3_key,
        "session_id": session_id,
        "status"    : "processing",
    }