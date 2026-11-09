import sqlite3
import json
import uuid
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from config import CHATS_DB_FILE

logger = logging.getLogger("aulaia.database")

def get_db() -> sqlite3.Connection:
    """Retorna una conexión a SQLite optimizada para FastAPI con WAL mode y Row factory."""
    CHATS_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHATS_DB_FILE), timeout=20.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_db():
    """Crea las tablas de conversaciones y mensajes si no existen."""
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conv_id ON messages(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
        """)
    logger.info("Base de datos SQLite de chats inicializada exitosamente.")


TITLE_PREFIXES = [
    "por favor", "oye", "hola", "explícame", "explicame", "explica", "muéstrame", "muestrame",
    "dime", "quiero saber", "podrías", "podrias", "puedes", "me puedes", "cuál es", "cual es",
    "cuáles son", "cuales son", "qué es", "que es", "qué son", "que son", "cómo se", "como se",
    "qué", "que", "cuál", "cual", "define", "genera", "dame",
    "la", "el", "los", "las", "un", "una", "de", "del",
]


def clean_title_from_prompt(prompt: str, max_chars: int = 42) -> str:
    """Genera un título corto y limpio (estilo Gemini) a partir de la primera pregunta del usuario."""
    cleaned = " ".join(prompt.split())
    cleaned = cleaned.strip("¿?¡!.,:; ")
    # Quitar muletillas y artículos iniciales de forma repetida ("¿Cuál es la definición..." -> "Definición...")
    changed = True
    while changed and cleaned:
        changed = False
        low = cleaned.lower()
        for prefix in TITLE_PREFIXES:
            if low.startswith(prefix + " ") or low.startswith(prefix + ","):
                cleaned = cleaned[len(prefix):].strip(" ,¿?¡!")
                changed = True
                break
    cleaned = cleaned.strip("¿?¡!.,:; ")
    if len(cleaned) < 3:
        cleaned = " ".join(prompt.split()).strip("¿?¡!.,:; ") or "Consulta Académica"
    if len(cleaned) > max_chars:
        cut = cleaned[:max_chars].rsplit(" ", 1)[0]
        cleaned = (cut or cleaned[:max_chars]).rstrip(" ,;:") + "…"
    # No capitalizar símbolos griegos (ε-clausura, δ, Σ...)
    if "Ͱ" <= cleaned[0] <= "Ͽ":
        return cleaned
    return cleaned[0].upper() + cleaned[1:]


def create_conversation(conv_id: Optional[str] = None, title: Optional[str] = None) -> Dict[str, Any]:
    """Crea una nueva conversación en SQLite."""
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cid = conv_id or str(uuid.uuid4())
    ctitle = title or "Nueva consulta"

    with get_db() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (cid, ctitle, now_iso, now_iso)
        )
        conn.commit()

    return {
        "id": cid,
        "title": ctitle,
        "created_at": now_iso,
        "updated_at": now_iso,
        "message_count": 0
    }


def list_conversations(limit: int = 50) -> List[Dict[str, Any]]:
    """Retorna la lista de conversaciones recientes con conteo de mensajes."""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "message_count": r["message_count"]
            }
            for r in rows
        ]


def get_conversation(conv_id: str) -> Optional[Dict[str, Any]]:
    """Retorna una conversación completa con todos sus mensajes formateados."""
    with get_db() as conn:
        conv_cur = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conv_id,)
        )
        conv = conv_cur.fetchone()
        if not conv:
            return None

        msg_cur = conn.execute("""
            SELECT id, role, content, sources, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
        """, (conv_id,))
        messages = []
        for m in msg_cur.fetchall():
            sources_val = None
            if m["sources"]:
                try:
                    sources_val = json.loads(m["sources"])
                except Exception:
                    sources_val = None
            messages.append({
                "id": m["id"],
                "role": m["role"],
                "content": m["content"],
                "sources": sources_val,
                "created_at": m["created_at"]
            })

        return {
            "id": conv["id"],
            "title": conv["title"],
            "created_at": conv["created_at"],
            "updated_at": conv["updated_at"],
            "messages": messages
        }


def add_message(
    conv_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Agrega un mensaje a la conversación.
    Si la conversación no existe, la crea.
    Si es el primer mensaje de usuario y el título es genérico, actualiza el título automáticamente.
    """
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sources_json = json.dumps(sources, ensure_ascii=False) if sources else None

    with get_db() as conn:
        # Verificar existencia de la conversación
        c_cur = conn.execute("SELECT id, title FROM conversations WHERE id = ?", (conv_id,))
        conv = c_cur.fetchone()

        if not conv:
            # Crear automáticamente
            auto_title = clean_title_from_prompt(content) if role == "user" else "Nueva consulta"
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conv_id, auto_title, now_iso, now_iso)
            )
        else:
            # Si el título actual es "Nueva consulta" y este es un mensaje de usuario, auto-titular
            if conv["title"] in ["Nueva consulta", "Consulta Académica"] and role == "user":
                new_title = clean_title_from_prompt(content)
                conn.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                    (new_title, now_iso, conv_id)
                )
            else:
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now_iso, conv_id)
                )

        # Insertar mensaje
        msg_cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
            (conv_id, role, content, sources_json, now_iso)
        )
        msg_id = msg_cur.lastrowid
        conn.commit()

    return {
        "id": msg_id,
        "conversation_id": conv_id,
        "role": role,
        "content": content,
        "sources": sources,
        "created_at": now_iso
    }


def get_recent_messages(conv_id: str, limit: int = 6) -> List[Dict[str, str]]:
    """Últimos `limit` mensajes de la conversación (en orden cronológico) para la memoria del chatbot."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (conv_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def delete_conversation(conv_id: str) -> bool:
    """Elimina una conversación y sus mensajes asociados."""
    with get_db() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        res = conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn.commit()
        return res.rowcount > 0


def rename_conversation(conv_id: str, new_title: str) -> bool:
    """Renombra el título de una conversación."""
    title = new_title.strip()
    if not title:
        return False
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        res = conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, now_iso, conv_id)
        )
        conn.commit()
        return res.rowcount > 0


def clear_all_conversations() -> bool:
    """Borra todo el historial."""
    with get_db() as conn:
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM conversations")
        conn.commit()
        return True


# Inicializar automáticamente al importar
init_db()

