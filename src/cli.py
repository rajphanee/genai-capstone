"""
Command line interface - handy for validating the system without the UI.

    python -m src info
    python -m src ingest sample_docs/*.*          # index one or more files
    python -m src list
    python -m src ask "How many PTO days do employees get?" [--trace]
    python -m src delete <doc_id>
    python -m src reset --yes                     # wipe the storage folder

Exit codes: 0 ok | 1 expected error (message printed) | 2 bad command usage
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from src.config import load_settings
from src.errors import DocQAError


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src", description="Enterprise Document Q&A")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    x = sub.add_parser("ingest")
    x.add_argument("files", nargs="+")
    sub.add_parser("list")
    x = sub.add_parser("ask")
    x.add_argument("question")
    x.add_argument("--trace", action="store_true", help="print every agent step")
    x = sub.add_parser("delete")
    x.add_argument("doc_id")
    x = sub.add_parser("reset")
    x.add_argument("--yes", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    try:
        if args.cmd == "reset":
            settings = load_settings()
            if not args.yes:
                print(f"This deletes {settings.data_dir}. Re-run with --yes.")
                return 1
            shutil.rmtree(settings.data_dir, ignore_errors=True)
            print(f"Deleted {settings.data_dir}")
            return 0

        from src.service import DocQAService     # imported here so 'reset' works even if ChromaDB is broken
        svc = DocQAService.create()

        if args.cmd == "info":
            print(json.dumps(svc.info(), indent=2))
        elif args.cmd == "ingest":
            failed = 0
            for name in args.files:
                path = Path(name)
                try:
                    if not path.is_file():
                        raise DocQAError(f"file not found: {path}")
                    r = svc.ingest(path.name, path.read_bytes())
                    print(f"OK    {path.name:28} {r['chunks']:>4} chunks  id={r['doc_id']}"
                          + ("  (already indexed)" if r["duplicate"] else ""))
                except DocQAError as exc:
                    failed += 1
                    print(f"FAIL  {path.name:28} {exc}", file=sys.stderr)
            return 1 if failed else 0
        elif args.cmd == "list":
            for d in svc.list_documents():
                print(f"{d['doc_id']}  {d['filename']:30} {d['chunks']:>4} chunks  {d['uploaded_at']}")
        elif args.cmd == "delete":
            svc.delete_document(args.doc_id)
            print(f"deleted {args.doc_id}")
        elif args.cmd == "ask":
            r = svc.ask(args.question)
            print(f"\n[{r['status'].upper()}] {r['answer']}\n")
            for c in r["citations"]:
                print(f"  [{c['tag']}] {c['filename']} - {c['location']} (relevance {c['relevance']:.2f})")
            if args.trace:
                print("\nAgent trace:")
                for s in r["trace"]:
                    print(f"  {s['agent']:10} {s['summary']}  ({s['ms']} ms)")
            print(f"\nreason: {r['reason']} | {r['total_ms']} ms")
        return 0
    except DocQAError as exc:
        print(f"ERROR [{exc.code}]: {exc}", file=sys.stderr)
        return 1
    except BrokenPipeError:      # output piped into `head`/`grep -q` - not an error
        return 0


if __name__ == "__main__":
    sys.exit(main())
