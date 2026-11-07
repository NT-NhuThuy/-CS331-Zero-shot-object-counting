"""Create a query-only bank from GT-free item manifests, without class labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import file_sha256
from .prepare_candidates import read_rows


def build(manifests: list[Path], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    terms = set()
    for manifest in manifests:
        for item in read_rows(manifest):
            query = item["query"]
            if not query or not query.endswith(" ."):
                raise ValueError("query is not in frozen canonical form")
            terms.add(query)
    with output.open("x") as stream:
        for query in sorted(terms):
            stream.write(json.dumps({"canonical_query":query,"aliases":[],
                                     "competitors":[],
                                     "query_source":"frozen_input_text_only"},
                                    sort_keys=True,ensure_ascii=False)+"\n")
    return {"queries":len(terms),"sha256":file_sha256(output),
            "competitor_terms":0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",action="append",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.manifest,args.out),sort_keys=True))


if __name__ == "__main__":
    main()
