import sys
import os

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "127.0.0.1")
    print("\n" + "=" * 55)
    print("  InsightPilot: Autonomous Research & Knowledge Agent")
    print(f"  Live Server running at: http://{host}:{port}")
    print(f"  API Documentation at:  http://{host}:{port}/docs")
    print("=" * 55 + "\n")
    uvicorn.run("backend.app:app", host=host, port=port, reload=True)
