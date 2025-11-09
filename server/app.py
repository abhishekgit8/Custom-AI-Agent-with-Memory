# app.py -- Memory Agent with Chroma persistence, migration, delete, clear, and HF router support
from fastapi import UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from datetime import datetime
import os
import uuid
import time
import json
import requests
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# OpenAI SDK (used to call HF router via OpenAI-compatible endpoint)
from openai import OpenAI

load_dotenv()

# --------------------
# Config
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")  # fine-grained HF token (Make calls to Inference Providers + Endpoints)
HF_MODEL = os.getenv("HF_MODEL", "gpt2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "4"))
CHROMA_INDEX_FILE = os.path.join(CHROMA_PERSIST_DIR, "chroma_index.json")

MIN_SAVE_LEN = int(os.getenv("MIN_SAVE_LEN", "20"))
MAX_MEMORIES_RETURN = int(os.getenv("MAX_MEMORIES_RETURN", str(TOP_K)))

# --------------------
# Initialize OpenAI-compatible client pointed at HF router (optional)
openai_client = None
if HF_API_TOKEN:
    try:
        openai_client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_API_TOKEN)
        print("OpenAI-compatible HF router client initialized.")
    except Exception as e:
        print("Could not initialize OpenAI client for HF router:", e)
else:
    print("HF_API_TOKEN not set — OpenAI-compatible HF router client not initialized.")

# --------------------
app = FastAPI(title="Memory Agent API (Chroma persistence)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Try optional imports
has_chromadb = False
has_sentence_transformers = False
try:
    import chromadb  # type: ignore
    from langchain.schema import Document
    from langchain.vectorstores import Chroma  # type: ignore
    has_chromadb = True
except Exception:
    has_chromadb = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import numpy as np  # type: ignore
    has_sentence_transformers = True
except Exception:
    has_sentence_transformers = False

# --------------------
# Embedding function (if sentence-transformers available)
embed_fn = None
embed_dim = None
if has_sentence_transformers:
    try:
        s2_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        def embed_fn(texts: List[str]) -> List[List[float]]:
            arr = s2_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return [list(map(float, vec)) for vec in arr]
        # lazy dimension inference
        try:
            embed_dim = len(embed_fn(["hello"])[0])
        except Exception:
            embed_dim = None
    except Exception:
        embed_fn = None
        has_sentence_transformers = False
else:
    embed_fn = None

# --------------------
# Ensure persist dir exists
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# --------------------
# Vector DB init: Chroma if available, otherwise in-memory fallback
use_in_memory = not has_chromadb
vectordb = None

if has_chromadb:
    try:
        if embed_fn is not None:
            vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embed_fn)
        else:
            vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR)
        use_in_memory = False
        print("\n✅ Using persistent Chroma vector database.")
        print(f"   Data directory: {CHROMA_PERSIST_DIR}")
        print("   Your memories will be saved and reloaded across restarts.\n")
    except Exception as e:
        import traceback
        print("\n⚠️  Chroma detected but initialization failed. Falling back to in-memory mode.")
        print("   Error:", repr(e))
        traceback.print_exc()
        use_in_memory = True

if use_in_memory:
    print("Using in-memory memory store (fallback). Install chromadb + sentence-transformers for persistence.")

    class InMemoryStore:
        def __init__(self):
            self.rows = []  # dicts: {id, text, metadata, embedding, created_at}
        def add(self, text: str, metadata: dict):
            doc_id = str(uuid.uuid4())
            emb = None
            if embed_fn is not None:
                try:
                    emb = embed_fn([text])[0]
                except Exception:
                    emb = None
            self.rows.append({"id": doc_id, "text": text, "metadata": metadata or {}, "embedding": emb, "created_at": time.time()})
            return doc_id
        def similarity_search(self, query: str, k: int=4):
            if embed_fn is not None and len(self.rows) and self.rows[0].get("embedding") is not None:
                try:
                    q_emb = embed_fn([query])[0]
                    import math
                    def cos(a,b):
                        da = math.sqrt(sum(x*x for x in a))
                        db = math.sqrt(sum(x*x for x in b))
                        if da == 0 or db == 0: return 0.0
                        return sum(x*y for x,y in zip(a,b))/(da*db)
                    scored = [(cos(q_emb, r["embedding"]), r) for r in self.rows if r.get("embedding") is not None]
                    scored.sort(key=lambda x: x[0], reverse=True)
                    return [r for s,r in scored[:k]]
                except Exception:
                    return list(reversed(self.rows))[:k]
            return list(reversed(self.rows))[:k]
        # def add_documents(self, docs, ids):
        #     for d,i in zip(docs, ids):
        #         self.add(d.page_content, d.metadata)
        def add_documents(self, docs, ids):
            """
            Compatibility wrapper for langchain-style calls.
            - Use the provided ids so we don't create duplicate entries with new UUIDs.
            - Skip insertion if the id already exists.
            - Optionally skip insertion if an identical text for the same user_id is already present.
            """
            for d, i in zip(docs, ids):
                # Extract text and metadata from Document-like or dict-like object.
                try:
                    text = getattr(d, "page_content", None) or (d.get("page_content") if isinstance(d, dict) else None)
                    meta = getattr(d, "metadata", None) or (d.get("metadata") if isinstance(d, dict) else {})
                except Exception:
                    text = str(d)
                    meta = {}

                doc_id = str(i)

                # If an entry with this id already exists, skip to avoid duplicate id entries.
                exists = any(r for r in self.rows if r.get("id") == doc_id)
                if exists:
                    # Optionally update the existing entry if you want to refresh timestamp/metadata:
                    # for r in self.rows:
                    #     if r.get("id") == doc_id:
                    #         r["text"] = text
                    #         r["metadata"] = meta or {}
                    #         r["created_at"] = time.time()
                    continue
                
                # Optional: cheap de-dup by exact text+user_id (prevents same text being inserted twice)
                try:
                    user_id = meta.get("user_id")
                    duplicate = None
                    for r in self.rows:
                        if user_id and r.get("metadata", {}).get("user_id") == user_id and r.get("text", "").strip() == (text or "").strip():
                            duplicate = r
                            break
                    if duplicate:
                        # if duplicate found, you can either skip or return existing id.
                        # We'll skip insertion to avoid duplicates.
                        continue
                except Exception:
                    pass
                
                # compute embedding if available
                emb = None
                if embed_fn is not None:
                    try:
                        emb = embed_fn([text])[0]
                    except Exception:
                        emb = None

                # insert using provided id
                self.rows.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": meta or {},
                    "embedding": emb,
                    "created_at": time.time()
                })

        def delete(self, ids: Optional[List[str]] = None, delete_all: bool = False):
            if delete_all:
                self.rows = []
                return
            if not ids:
                return
            self.rows = [r for r in self.rows if r["id"] not in ids]
        def persist(self):
            pass
    vectordb = InMemoryStore()

# --------------------
# Simple local index helpers for Chroma listing/deleting
def _load_chroma_index():
    try:
        if os.path.exists(CHROMA_INDEX_FILE):
            with open(CHROMA_INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_chroma_index(index: dict):
    try:
        with open(CHROMA_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Could not save chroma index file:", e)

def _index_add(doc_id: str, text: str, metadata: dict):
    idx = _load_chroma_index()
    idx[doc_id] = {"text": text, "metadata": metadata, "created_at": time.time()}
    _save_chroma_index(idx)

def _index_delete(doc_id: str):
    idx = _load_chroma_index()
    if doc_id in idx:
        del idx[doc_id]
        _save_chroma_index(idx)

def _index_clear():
    _save_chroma_index({})

# --------------------
# Core helpers
def store_memory(user_id: str, text: str, metadata: dict | None = None, doc_id: Optional[str] = None):
    """
    Store memory with optional provided doc_id. Returns stored record dict:
      {id, text, metadata, created_at}
    Skips insertion if duplicate (same user_id and identical text).
    """
    metadata = metadata or {}
    doc_id = doc_id or str(uuid.uuid4())
    created_at = time.time()

    # cheap dedupe: exact match for the same user
    try:
        if hasattr(vectordb, "rows"):
            for r in vectordb.rows:
                if r.get("metadata", {}).get("user_id") == user_id and r.get("text", "").strip() == text.strip():
                    return {"id": r["id"], "text": r["text"], "metadata": r["metadata"], "created_at": r.get("created_at")}
    except Exception:
        pass

    # Try Chroma-style add_documents if available
    try:
        if hasattr(vectordb, "add_documents"):
            try:
                from langchain.schema import Document
                doc = Document(page_content=text, metadata={"user_id": user_id, "created_at": created_at, **metadata})
                vectordb.add_documents([doc], ids=[doc_id])
                try:
                    vectordb.persist()
                except Exception:
                    pass
            except Exception:
                if hasattr(vectordb, "add"):
                    doc_id = vectordb.add(text, {"user_id": user_id, **metadata})
            _index_add(doc_id, text, {"user_id": user_id, **metadata})
            return {"id": doc_id, "text": text, "metadata": {"user_id": user_id, **metadata}, "created_at": created_at}
    except Exception as e:
        print("store_memory chroma path failed:", e)

    # fallback to in-memory
    try:
        new_id = vectordb.add(text, {"user_id": user_id, **metadata})
        return {"id": new_id, "text": text, "metadata": {"user_id": user_id, **metadata}, "created_at": created_at}
    except Exception:
        return {"id": doc_id, "text": text, "metadata": {"user_id": user_id, **metadata}, "created_at": created_at}

def retrieve_memories(query: str, k: int = TOP_K):
    """
    Returns list of memory dicts: [{"id","text","metadata","created_at"}, ...]
    """
    if not query or not query.strip():
        return []
    try:
        if hasattr(vectordb, "similarity_search"):
            results = vectordb.similarity_search(query, k=k)
            out = []
            for r in results:
                if hasattr(r, "page_content"):
                    text = r.page_content
                    meta = getattr(r, "metadata", {}) or {}
                    rid = getattr(r, "id", None) or meta.get("id") or str(uuid.uuid4())
                    out.append({"id": rid, "text": text, "metadata": meta, "created_at": meta.get("created_at")})
                elif isinstance(r, dict):
                    out.append({"id": r.get("id"), "text": r.get("text"), "metadata": r.get("metadata", {}), "created_at": r.get("created_at")})
                else:
                    out.append({"id": None, "text": str(r), "metadata": {}, "created_at": None})
            return out[:k]
    except Exception:
        pass
    return []

# --------------------
# HF inference (OpenAI-compatible client first, fallback to router POST)
def call_hf_inference(prompt: str, max_new_tokens: int = 256) -> str:
    # Prefer OpenAI-compatible client (chat completions)
    if openai_client is not None:
        try:
            model_to_use = os.getenv("HF_MODEL") or HF_MODEL
            resp = openai_client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
            )
            choice = resp.choices[0]
            # newer SDK returns message object
            if hasattr(choice, "message") and getattr(choice.message, "content", None):
                return choice.message.content
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
            if "text" in choice:
                return choice["text"]
            return str(resp)
        except Exception as e:
            return f"[HF router (openai client) error] {e}"

    # Fallback: older router POST for models that accept plain generation
    if not HF_API_TOKEN:
        return "HF_API_TOKEN not set in server environment. Responses disabled."
    base_router = "https://router.huggingface.co/hf-inference/models"
    model = HF_MODEL
    url = f"{base_router}/{model}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens}}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        if isinstance(data, dict) and "error" in data:
            return f"[HF error] {data['error']}"
        return str(data)
    except Exception as e:
        return f"[HF inference error] {e}"

# --------------------
# Pydantic models
class ChatRequest(BaseModel):
    user_id: str
    message: str
    save_memory: Optional[bool] = True

class ChatResponse(BaseModel):
    reply: str
    used_memories: List[str]

class MemoryIn(BaseModel):
    user_id: str
    text: str

# --------------------
# Routes
@app.get("/api/health")
def health():
    return {"status": "ok", "use_chromadb": has_chromadb, "use_sentence_transformers": has_sentence_transformers}

@app.post("/api/memory/add")
def add_memory(payload: MemoryIn):
    """
    Accept MemoryIn and return canonical inserted id string.
    store_memory may return either a dict (with "id") or a plain id string.
    """
    result = store_memory(payload.user_id, payload.text, {"source": "manual"})
    # If store_memory returned a dict, extract id
    if isinstance(result, dict):
        doc_id = result.get("id")
    else:
        doc_id = str(result)
    return {"status": "ok", "id": doc_id}


from fastapi import Query

@app.get("/api/memory/list")
def list_memories(limit: int = Query(100, le=1000), offset: int = 0):
    items_by_id = {}
    try:
        if hasattr(vectordb, "rows"):
            for r in vectordb.rows:
                items_by_id[r["id"]] = {"id": r["id"], "text": r["text"], "metadata": r["metadata"], "created_at": r.get("created_at")}
    except Exception:
        pass
    try:
        idx = _load_chroma_index()
        for k, v in idx.items():
            entry = {"id": k, "text": v.get("text"), "metadata": v.get("metadata"), "created_at": v.get("created_at")}
            cur = items_by_id.get(k)
            if not cur or (entry.get("created_at") and (not cur.get("created_at") or entry["created_at"] > cur["created_at"])):
                items_by_id[k] = entry
    except Exception:
        pass

    items = list(items_by_id.values())
    items.sort(key=lambda x: x.get("created_at") or 0, reverse=True)
    sliced = items[offset: offset + limit]
    return {"count": len(items), "items": sliced}


@app.post("/api/memory/migrate")
def migrate_memories():
    """
    Move existing in-memory rows into Chroma (persistent store).
    """
    if not has_chromadb:
        raise HTTPException(status_code=400, detail="Chroma not installed. Install chromadb and restart.")
    migrated = 0
    # find in-memory rows (if vectordb is still the in-memory wrapper)
    try:
        if hasattr(vectordb, "rows"):
            rows = list(vectordb.rows)
            for r in rows:
                try:
                    store_memory(r["metadata"].get("user_id", "unknown"), r["text"], r["metadata"])
                    migrated += 1
                except Exception:
                    pass
            # clear in-memory after migration
            try:
                vectordb.delete(delete_all=True)
            except Exception:
                if hasattr(vectordb, "rows"):
                    vectordb.rows = []
    except Exception:
        pass
    return {"migrated": migrated}

@app.delete("/api/memory/delete/{memory_id}")
def delete_memory(memory_id: str):
    # delete from in-memory if present
    try:
        if hasattr(vectordb, "rows"):
            vectordb.delete(ids=[memory_id])
    except Exception:
        pass
    # delete from local index and try vectordb delete APIs
    _index_delete(memory_id)
    try:
        if hasattr(vectordb, "delete"):
            try:
                vectordb.delete(ids=[memory_id])
            except TypeError:
                try:
                    vectordb.delete(ids=[memory_id], delete_all=False)
                except Exception:
                    pass
    except Exception:
        pass
    return {"status": "ok", "id": memory_id}

@app.delete("/api/memory/clear")
def clear_memories():
    try:
        if hasattr(vectordb, "delete"):
            try:
                vectordb.delete(delete_all=True)
            except TypeError:
                try:
                    vectordb.delete(ids=None, delete_all=True)
                except Exception:
                    pass
        if hasattr(vectordb, "rows"):
            vectordb.rows = []
    except Exception:
        pass
    _index_clear()
    # remove index file only (safe)
    try:
        if os.path.exists(CHROMA_INDEX_FILE):
            os.remove(CHROMA_INDEX_FILE)
    except Exception:
        pass
    return {"status": "ok"}

class ChatResponseV2(BaseModel):
    reply: str
    used_memories: List[dict]  # list of memory dicts

@app.post("/api/chat", response_model=ChatResponseV2)
async def chat(payload: ChatRequest):
    user_id = payload.user_id
    message = payload.message or ""
    used = retrieve_memories(message, k=TOP_K)

    system = "You are a helpful assistant. Use the memories to personalize answers."
    mem_block = "\n".join(f"- [{m.get('id')}] {m.get('text')}" for m in used) if used else "No saved memories."
    prompt = f"{system}\n\nUser message: {message}\n\nRelevant memories:\n{mem_block}\n\nAssistant response:"

    # call blocking HF client in thread if necessary (your call_hf_inference is sync)
    from concurrent.futures import ThreadPoolExecutor
    loop = None
    try:
        import asyncio
        loop = asyncio.get_event_loop()
    except Exception:
        loop = None

    def sync_call():
        return call_hf_inference(prompt)

    reply = None
    if loop and loop.is_running():
        # run in executor to avoid blocking event loop
        with ThreadPoolExecutor() as ex:
            reply = await loop.run_in_executor(ex, sync_call)
    else:
        reply = sync_call()

    # Save memory only if user asked and message long enough
    if payload.save_memory and len(message.strip()) >= MIN_SAVE_LEN:
        # Save asynchronously in background thread/process to avoid delaying response
        try:
            store_memory(user_id, message, {"source": "user_message"})
        except Exception:
            pass

    return ChatResponseV2(reply=reply, used_memories=used)

from fastapi import Query

@app.post("/api/voice/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    user_id: Optional[str] = Query(None, description="Optional user id to associate memory/chat"),
    save_memory: bool = Query(True, description="Whether to save the transcription as a memory"),
    run_chat: bool = Query(True, description="Whether to send the transcription to the LLM for a reply"),
):
    """
    Upload audio -> transcribe -> (optionally) send transcription to model for generation.

    Returns JSON:
      {
        "transcription": "...",
        "reply": "...",                # only present if run_chat is true
        "used_memories": [{...}, ...]  # model-used memories (if any)
      }
    """
    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        return JSONResponse({"error": "File too large (max 20MB)"}, status_code=413)

    filename = getattr(file, "filename", "audio_input")
    content_type = getattr(file, "content_type", "application/octet-stream")

    transcription_text = None

    # 1) Try openai_client if present
    if openai_client is not None:
        try:
            from io import BytesIO
            bio = BytesIO(contents)
            # SDKs vary; try file-like first, then bytes
            try:
                resp = openai_client.audio.transcriptions.create(file=bio, model=os.getenv("WHISPER_MODEL", "whisper-1"))
            except Exception:
                resp = openai_client.audio.transcriptions.create(file=contents, model=os.getenv("WHISPER_MODEL", "whisper-1"))

            if hasattr(resp, "text"):
                transcription_text = resp.text
            elif isinstance(resp, dict):
                transcription_text = resp.get("text") or resp.get("transcription") or str(resp)
            else:
                transcription_text = str(resp)
        except Exception:
            transcription_text = None

    # 2) HF multipart fallback
    if not transcription_text and HF_API_TOKEN:
        try:
            url = f"https://api-inference.huggingface.co/models/{os.getenv('WHISPER_MODEL_HF', 'openai/whisper-large-v2')}"
            headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
            files = {"file": (filename, contents, content_type)}
            resp = requests.post(url, headers=headers, files=files, timeout=120)
            if resp.ok:
                data = resp.json()
                # try a few common keys/shapes
                if isinstance(data, dict):
                    transcription_text = data.get("text") or data.get("transcription") or data.get("recognized_text")
                    if not transcription_text and "segments" in data and isinstance(data["segments"], list):
                        transcription_text = " ".join(seg.get("text","") for seg in data["segments"]).strip()
                else:
                    transcription_text = str(data)
            else:
                return JSONResponse({"error": "HF inference returned an error", "details": resp.text}, status_code=502)
        except Exception as e:
            return JSONResponse({"error": f"transcription failed: {e}"}, status_code=500)

    if not transcription_text:
        return JSONResponse({"error": "No transcription available (no client configured or transcription failed)."}, status_code=400)

    # Optionally save the transcription as a memory
    if save_memory and user_id:
        try:
            store_memory(user_id, transcription_text, {"source": "audio_transcript"})
        except Exception:
            pass

    result = {"transcription": transcription_text}

    # Optionally send transcription to LLM (model) with memories
    if run_chat:
        # Retrieve memories relevant to transcription_text
        used = retrieve_memories(transcription_text, k=TOP_K)
        # Build prompt (same style as /api/chat)
        system = "You are a helpful assistant. Use the memories to personalize answers."
        mem_block = "\n".join(f"- [{m.get('id')}] {m.get('text')}" for m in used) if used else "No saved memories."
        prompt = f"{system}\n\nUser message (from audio): {transcription_text}\n\nRelevant memories:\n{mem_block}\n\nAssistant response:"

        # call the blocking inference function in executor to avoid blocking event loop
        from concurrent.futures import ThreadPoolExecutor
        import asyncio
        loop = asyncio.get_event_loop()
        def sync_call():
            return call_hf_inference(prompt)
        reply = None
        if loop and loop.is_running():
            with ThreadPoolExecutor() as ex:
                reply = await loop.run_in_executor(ex, sync_call)
        else:
            reply = sync_call()

        result["reply"] = reply
        result["used_memories"] = used

    return result
