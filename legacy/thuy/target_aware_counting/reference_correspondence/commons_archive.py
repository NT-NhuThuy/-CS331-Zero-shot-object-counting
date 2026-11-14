"""Archive a query-only Wikimedia Commons search as a local reference snapshot.

No evaluation labels, result scores, or image-specific classes are used. Search
order is fixed by the provider. Downloaded images are content addressed; a
fragment is committed only after every rank for one query has been processed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .config import file_sha256, stable_hash
from .reference_bank import _read_query_bank


API = "https://commons.wikimedia.org/w/api.php"
MIME_EXT = {"image/jpeg":"jpg", "image/png":"png", "image/webp":"webp"}


def _request(url: str, user_agent: str, max_bytes: int) -> bytes:
    host = urllib.parse.urlparse(url).hostname
    if host not in {"commons.wikimedia.org","upload.wikimedia.org","thumb.wikimedia.org"}:
        raise ValueError(f"unexpected provider image host: {host}")
    request = urllib.request.Request(url,headers={"User-Agent":user_agent})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request,timeout=45) as response:
                data = response.read(max_bytes+1)
            if len(data) > max_bytes:
                raise ValueError("provider image exceeds locked byte budget")
            return data
        except urllib.error.HTTPError as error:
            if error.code == 429:
                raise RuntimeError("provider_rate_limited_429; stop and resume later") from error
            if attempt == 2 or error.code not in (500,502,503,504):
                raise
            time.sleep(1+attempt)
        except (TimeoutError,urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(1+attempt)
    raise AssertionError("unreachable")


def _search(term: str, limit: int, thumb_width: int, user_agent: str) -> list[dict]:
    params = {"action":"query","generator":"search","gsrsearch":term,
              "gsrnamespace":6,"gsrlimit":limit,"prop":"imageinfo",
              "iiprop":"url|extmetadata|mime","iiurlwidth":thumb_width,
              "format":"json","formatversion":2}
    for attempt in range(5):
        data = _request(API+"?"+urllib.parse.urlencode(params),user_agent,8*1024*1024)
        result = json.loads(data)
        if "error" not in result:
            return sorted(result.get("query",{}).get("pages",[]),
                          key=lambda p:p.get("index",10**9))
        if result["error"].get("code") != "cirrussearch-too-busy-error" or attempt == 4:
            raise RuntimeError(f"Commons search error: {result['error']}")
        time.sleep(2**(attempt+1))
    raise AssertionError("unreachable")


def _terms(bank: dict[str,dict]) -> list[tuple[str,str,str]]:
    terms = []
    for query,record in sorted(bank.items()):
        for term in [query,*record.get("aliases",[])]:
            terms.append((query,"target",term.strip(" .")))
        for competitor in record.get("competitors",[]):
            terms.append((query,"competitor",competitor["term"].strip(" .")))
    if len(set(terms)) != len(terms):
        raise ValueError("duplicate query/role/retrieval term")
    return terms


def _store_image(data: bytes, mime: str, root: Path, minimum_edge: int) -> tuple[Path,str]:
    if mime not in MIME_EXT:
        raise ValueError("unsupported image MIME")
    with Image.open(io.BytesIO(data)) as opened:
        opened.verify()
    with Image.open(io.BytesIO(data)) as opened:
        if min(opened.size) < minimum_edge:
            raise ValueError("provider image too small")
    digest = hashlib.sha256(data).hexdigest()
    path = root/"images"/f"{digest}.{MIME_EXT[mime]}"
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if file_sha256(path) != digest:
            raise ValueError("content addressed image hash mismatch")
        return path,digest
    temporary = path.with_name(path.name+".partial")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)
    return path,digest


def archive(query_bank: Path, root: Path, max_rank: int, thumb_width: int,
            minimum_edge: int, max_image_bytes: int, user_agent: str,
            max_new_terms: int | None, provider_out: Path | None = None,
            allow_partial: bool = False) -> dict:
    if not user_agent.strip() or max_rank < 1 or thumb_width < 128 or \
            minimum_edge < 1 or max_image_bytes < 1024:
        raise ValueError("archive policy is incomplete")
    if max_new_terms is not None and max_new_terms < 0:
        raise ValueError("max_new_terms must be nonnegative")
    bank = _read_query_bank(query_bank)
    terms = _terms(bank)
    (root/"fragments").mkdir(parents=True,exist_ok=True)
    new_terms = 0
    for query,role,term in terms:
        key = stable_hash([query,role,term])[:24]
        fragment = root/"fragments"/f"{key}.json"
        if fragment.exists():
            existing = json.loads(fragment.read_text())
            if existing["query_bank_sha256"] != file_sha256(query_bank) or \
                    existing["search_policy"] != [max_rank,thumb_width,minimum_edge,max_image_bytes]:
                raise ValueError("resume fragment has changed query bank/search policy")
            continue
        if max_new_terms is not None and new_terms >= max_new_terms:
            break
        retrieved_at = datetime.now(timezone.utc).isoformat()
        rows = []
        for page in _search(term,max_rank,thumb_width,user_agent):
            rank = int(page["index"])
            info = page.get("imageinfo",[{}])[0] if page.get("imageinfo") else {}
            metadata = info.get("extmetadata",{})
            license_name = metadata.get("LicenseShortName",{}).get("value","")
            license_url = metadata.get("LicenseUrl",{}).get("value","")
            image_url = info.get("thumburl") or info.get("url","")
            mime = info.get("mime","")
            row = {"canonical_query":query,"semantic_role":role,"retrieval_term":term,
                   "retrieval_rank":rank,"source_url":info.get("descriptionurl",""),
                   "original_url":info.get("url",""),"download_url":image_url,
                   "retrieved_at":retrieved_at,"license":license_name or "unknown",
                   "license_url":license_url,"provider":"Wikimedia Commons MediaWiki search API",
                   "provider_page_id":page.get("pageid"),"provider_title":page.get("title"),
                   "image_mime":mime,"image_path":str(root/"broken"/f"{key}_{rank}.img")}
            try:
                if not image_url or not row["source_url"] or not license_name:
                    raise ValueError("missing image URL, source page, or license")
                data = _request(image_url,user_agent,max_image_bytes)
                path,digest = _store_image(data,mime,root,minimum_edge)
                row["image_path"] = str(path)
                row["download_sha256"] = digest
                row["download_status"] = "ok"
            except (OSError,ValueError,urllib.error.URLError) as error:
                row["download_status"] = "invalid"
                row["download_error"] = type(error).__name__+": "+str(error)
            rows.append(row)
        record = {"query_bank_sha256":file_sha256(query_bank),
                  "search_policy":[max_rank,thumb_width,minimum_edge,max_image_bytes],
                  "query":query,"role":role,"term":term,"rows":rows}
        with fragment.open("x") as stream:
            json.dump(record,stream,sort_keys=True,ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        new_terms += 1
        print(json.dumps({"query":query,"role":role,"rows":len(rows),
                          "downloaded":sum(r["download_status"] == "ok" for r in rows),
                          "new_terms":new_terms}),flush=True)
        time.sleep(1)
    complete = True
    provider_rows = []
    for query,role,term in terms:
        fragment = root/"fragments"/f"{stable_hash([query,role,term])[:24]}.json"
        if not fragment.exists():
            complete = False
            continue
        provider_rows.extend(json.loads(fragment.read_text())["rows"])
    if (complete or allow_partial) and provider_out is not None:
        if provider_out.exists():
            raise FileExistsError(provider_out)
        with provider_out.open("x") as stream:
            for row in provider_rows:
                stream.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
    return {"complete":complete,"terms":len(terms),"new_terms":new_terms,
            "provider_rows":len(provider_rows),
            "downloaded":sum(r["download_status"] == "ok" for r in provider_rows),
            "provider_sha256":file_sha256(provider_out)
            if provider_out and (complete or allow_partial) else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-bank",type=Path,required=True)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--max-rank",type=int,required=True)
    parser.add_argument("--thumb-width",type=int,required=True)
    parser.add_argument("--minimum-edge",type=int,required=True)
    parser.add_argument("--max-image-bytes",type=int,required=True)
    parser.add_argument("--user-agent",required=True)
    parser.add_argument("--max-new-terms",type=int)
    parser.add_argument("--provider-out",type=Path)
    parser.add_argument("--allow-partial",action="store_true",
                        help="write an explicitly incomplete provider manifest for technical smoke only")
    args = parser.parse_args()
    print(json.dumps(archive(args.query_bank,args.root,args.max_rank,args.thumb_width,
                             args.minimum_edge,args.max_image_bytes,args.user_agent,
                             args.max_new_terms,args.provider_out,args.allow_partial),sort_keys=True))


if __name__ == "__main__":
    main()
