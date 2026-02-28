from __future__ import annotations

import argparse

import uvicorn


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the PALA probe web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "tools.probe_web.app:app",
        host=str(args.host),
        port=int(args.port),
        reload=bool(args.reload),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
