#!/usr/bin/env python3
"""
Batch reindexer for RAG ingest: repeatedly calls the backend endpoint
POST /api/v1/ingest/reindex_missing until no candidates remain.

Default behavior: process ALL documents (not only missing), in batches of 20.

Usage examples:
  python3 scripts/reindex_all.py
  python3 scripts/reindex_all.py --base http://localhost:8000 --limit 20 --sleep 1.5
  python3 scripts/reindex_all.py --only-missing true

Exit code is non-zero on unrecoverable HTTP errors.
"""

import argparse
import sys
import time
from typing import Any, Dict

try:
    import httpx
except Exception as e:
    print("Please install httpx (pip install httpx)")
    raise


def post_json(client: httpx.Client, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = client.post(url, json=payload)
    r.raise_for_status()
    return r.json()


def main():
    parser = argparse.ArgumentParser(description="Batch reindex all documents via /ingest/reindex_missing")
    parser.add_argument("--base", default="http://localhost:8000", help="Base URL of the API, default: http://localhost:8000")
    parser.add_argument("--limit", type=int, default=20, help="Batch size per request (limit param), default: 20")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep seconds between batches, default: 1.0")
    parser.add_argument("--only-missing", dest="only_missing", default=False, choices=[True, False], type=lambda x: str(x).lower() in ["1","true","t","yes","y"], help="Only process missing docs (chunks==0). Default: false (process all)")
    parser.add_argument("--max-loops", type=int, default=1000, help="Safety stop after N loops, default: 1000")
    parser.add_argument("--timeout", type=float, default=300.0, help="HTTP timeout seconds per request, default: 300")
    parser.add_argument("--adaptive", dest="adaptive", default=True, choices=[True, False], type=lambda x: str(x).lower() in ["1","true","t","yes","y"], help="Adaptively reduce batch size on timeout, default: true")
    parser.add_argument("--min-batch", type=int, default=1, help="Minimum batch size when adapting, default: 1")
    args = parser.parse_args()

    base = args.base.rstrip("/")
    dry_url = f"{base}/api/v1/ingest/reindex_missing"
    exec_url = f"{base}/api/v1/ingest/reindex_missing"

    total_processed = 0
    total_success = 0
    total_errors = 0
    loop = 0

    print(f"Starting batch reindex: base={base} limit={args.limit} only_missing={args.only_missing} timeout={args.timeout}s adaptive={args.adaptive}")

    try:
        with httpx.Client(timeout=httpx.Timeout(args.timeout)) as client:
            while True:
                loop += 1
                if loop > args.max_loops:
                    print(f"Reached max loops ({args.max_loops}). Stopping.")
                    break

                # Dry run to check remaining
                try:
                    dry = post_json(client, dry_url, {"dry_run": True, "only_missing": args.only_missing})
                except httpx.HTTPStatusError as he:
                    print(f"[ERROR] dry_run HTTP {he.response.status_code}: {he.response.text[:300]}")
                    return 2
                except Exception as e:
                    print(f"[ERROR] dry_run failed: {e}")
                    return 2

                to_process = int(dry.get("to_process") or 0)
                print(f"Remaining candidates: {to_process}")
                if to_process <= 0:
                    print("No more candidates. Done.")
                    break

                batch = min(args.limit, to_process)
                print(f"Processing batch size: {batch}")

                # Execute batch
                exec_payload = {"limit": batch, "only_missing": args.only_missing}
                while True:
                    try:
                        resp = post_json(client, exec_url, exec_payload)
                        break
                    except httpx.TimeoutException:
                        if args.adaptive and batch > args.min_batch:
                            # Reduce batch and retry
                            new_batch = max(args.min_batch, max(1, batch // 2))
                            if new_batch == batch:
                                print(f"[WARN] timeout at batch={batch}, cannot reduce further (min_batch={args.min_batch}).")
                                return 3
                            print(f"[WARN] timeout at batch={batch}, reducing to {new_batch} and retrying...")
                            batch = new_batch
                            exec_payload = {"limit": batch, "only_missing": args.only_missing}
                            continue
                        else:
                            print(f"[ERROR] exec timed out at batch={batch}")
                            return 3
                    except httpx.HTTPStatusError as he:
                        print(f"[ERROR] exec HTTP {he.response.status_code}: {he.response.text[:500]}")
                        return 3
                    except Exception as e:
                        print(f"[ERROR] exec failed: {e}")
                        return 3

                processed = int(resp.get("processed") or 0)
                successes = int(resp.get("successes") or 0)
                errors = resp.get("errors") or []

                total_processed += processed
                total_success += successes
                total_errors += len(errors)

                print(f"Batch processed={processed} successes={successes} errors={len(errors)}")
                if errors:
                    for e in errors[:5]:
                        print(f"  error id={e.get('id')} title={e.get('title')} err={str(e.get('error'))[:200]}")

                time.sleep(args.sleep)

    except KeyboardInterrupt:
        print("Interrupted by user.")

    print("\n=== Summary ===")
    print(f"Total processed: {total_processed}")
    print(f"Total successes: {total_success}")
    print(f"Total errors:    {total_errors}")

    # Optional: return non-zero if any errors occurred
    if total_errors > 0:
        sys.exit(4)


if __name__ == "__main__":
    main()
