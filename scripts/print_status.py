#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Ensure project root on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from app.deps import SessionLocal, milvus
from sqlalchemy import text as sql_text


def main():
    db = SessionLocal()
    try:
        total_chunks = db.execute(sql_text("SELECT COUNT(*) FROM doc_chunks")).scalar()
        docs = db.execute(sql_text("SELECT COUNT(*) FROM documents" )).scalar()
    finally:
        db.close()
    stats = milvus.get_collection_stats("kb_chunks") if milvus else {}
    print({
        'documents': int(docs or 0),
        'mysql_doc_chunks': int(total_chunks or 0),
        'milvus_stats': stats,
    })


if __name__ == "__main__":
    main()
