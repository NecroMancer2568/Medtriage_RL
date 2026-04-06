"""
server/app.py — Entry point for openenv validate compatibility.
Wraps the existing FastAPI app in src/server.py.
"""

import uvicorn
from src.server import app


def main():
    """Main entry point for openenv-core compatibility."""
    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()