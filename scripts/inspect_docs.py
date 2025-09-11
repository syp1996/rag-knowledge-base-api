import json
import sys
from pathlib import Path

def main():
    for pth in sys.argv[1:]:
        p = Path(pth)
        if not p.exists():
            print(f"missing: {pth}")
            continue
        try:
            j = json.loads(p.read_text())
        except Exception as e:
            print(f"failed to read {pth}: {e}")
            continue
        content = j.get('content')
        title = j.get('title')
        if isinstance(content, dict):
            keys = list(content.keys())
        else:
            keys = type(content).__name__
        print(f"file={pth} title={title} content_keys={keys}")

if __name__ == "__main__":
    main()

