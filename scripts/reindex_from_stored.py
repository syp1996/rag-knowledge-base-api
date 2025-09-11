import sys
import json
import httpx

BASE = "http://localhost:8000"

def reindex(doc_id: int):
    with httpx.Client(timeout=30.0) as client:
        r = client.get(f"{BASE}/api/documents/{doc_id}")
        r.raise_for_status()
        doc = r.json()
        content = doc.get("content")
        if not content:
            print(f"doc {doc_id}: no content to reindex")
            return
        payload = {"content": content}
        r2 = client.put(f"{BASE}/api/v1/ingest/text/{doc_id}", json=payload)
        try:
            r2.raise_for_status()
        except Exception:
            print(r2.text)
            raise
        print(f"doc {doc_id} reindexed: {r2.json()}")

def main():
    ids = [int(x) for x in sys.argv[1:]]
    for doc_id in ids:
        reindex(doc_id)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/reindex_from_stored.py <doc_id> [<doc_id> ...]")
        sys.exit(1)
    main()

