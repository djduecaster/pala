from __future__ import annotations

import argparse

import uvicorn

from .app import create_app



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Web review app for ft_capture datasets")
    parser.add_argument("--dataset-root", default="logs/ft_capture")
    parser.add_argument("--catalog", default="config/ft_scenarios.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--mount-prefix", default="/dataset-files")
    return parser



def main() -> int:
    args = _build_parser().parse_args()
    app = create_app(
        dataset_root=args.dataset_root,
        catalog_path=args.catalog,
        mount_prefix=args.mount_prefix,
    )
    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
