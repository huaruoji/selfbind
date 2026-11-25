#!/usr/bin/env python3
"""Run independent LLM annotators over the Identity Scope Evaluation Suite.

Each annotator receives the released codebook as a system prompt and labels one
prompt at a time, returning strict JSON. Outputs are written as annotation
sheets in the same format the human annotators use, so the existing scorer
(experiments/score_human_annotation.py) consumes both without changes.

Usage:
  python experiments/llm_annotate_scope.py --models deepseek gpt gemini
  python experiments/llm_annotate_scope.py --models deepseek --limit 5   # smoke test
  python experiments/llm_annotate_scope.py --report                      # agreement only
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "annotation_kit"
CRED = Path.home() / ".dsh" / ".credentials.yaml"
ENDPOINT = "https://www.cun.ai/v1/chat/completions"

# Relay ids verified 2026-09-20 by probing /v1/chat/completions. gemini-3.8-flash
# and gemini-3.7-flash are advertised in /v1/models but return 503 with no
# available channel, so they are deliberately not used here.
ANNOTATORS: dict[str, str] = {
    "deepseek": "DeepSeek-V4.1-Flash",
    "gpt": "gpt-5.6-luna",
    "gemini": "gemini-3.6-flash",
}

SYSTEM_TEMPLATE = (
    "You are an independent annotator for an NLP evaluation suite. Apply the annotation "
    "rule below exactly as written and label the single prompt you are given. Judge the "
    "prompt only. Do not speculate about what a model would answer.\n\n"
    "Reply with strict JSON and nothing else, in this exact shape:\n"
    '{{"scope": <1 or 0>, "confidence": <1, 2 or 3>, "basis": "<one short sentence>"}}\n\n'
    "CODEBOOK\n========\n{codebook}\n"
)


def load_api_key() -> str:
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML is required to read ~/.dsh/.credentials.yaml")
    data = yaml.safe_load(CRED.read_text())
    key = (data.get("refs") or {}).get("CUNAI_API_KEY")
    if not key:
        raise SystemExit("CUNAI_API_KEY not found in ~/.dsh/.credentials.yaml")
    return key


def parse_label(content: str) -> dict | None:
    """Pull the JSON object out of a model reply, tolerating code fences."""
    if not content:
        return None
    text = content.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    scope = obj.get("scope")
    if isinstance(scope, str) and scope.strip() in {"0", "1"}:
        scope = int(scope.strip())
    if scope not in (0, 1):
        return None
    conf = obj.get("confidence", 2)
    try:
        conf = max(1, min(3, int(conf)))
    except (TypeError, ValueError):
        conf = 2
    return {"scope": scope, "confidence": conf, "basis": str(obj.get("basis", ""))[:300]}


def call_model(key: str, model: str, system: str, user: str,
               retries: int = 4, timeout: int = 180) -> tuple[dict | None, dict]:
    body = json.dumps({
        "model": model,
        "max_tokens": 1500,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }).encode()
    last_err = ""
    for attempt in range(retries):
        req = urllib.request.Request(
            ENDPOINT, data=body,
            headers={"Authorization": f"Bearer {key}",
                     "User-Agent": "CUN.AI-DSH/1.0",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read())
            usage = payload.get("usage", {})
            content = payload["choices"][0]["message"].get("content", "")
            return parse_label(content), usage
        except urllib.error.HTTPError as exc:
            last_err = f"HTTP {exc.code}"
            detail = exc.read()[:200].decode(errors="replace")
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep(3 * (attempt + 1))
                continue
            return None, {"_error": f"{last_err} {detail}"}
        except Exception as exc:  # noqa: BLE001 - network flakiness is expected
            last_err = str(exc)[:120]
            time.sleep(3 * (attempt + 1))
    return None, {"_error": last_err}


def annotate(model_key: str, model_id: str, key: str, items: list[dict],
             out_path: Path) -> dict:
    system = SYSTEM_TEMPLATE.format(codebook=(KIT / "CODEBOOK.md").read_text())
    rows: list[dict] = []
    failures: list[dict] = []
    toks_in = toks_out = 0
    start = time.time()

    for i, item in enumerate(items, 1):
        raw, usage = call_model(key, model_id, system,
                                "PROMPT TO LABEL:\n" + item["text"])
        if raw is None:
            failures.append({"item_id": item["item_id"], "error": usage.get("_error", "unparsed")})
            rows.append({"item_id": item["item_id"], "scope_label": "",
                         "confidence_1to3": "", "notes": "LLM_FAILED"})
        else:
            rows.append({"item_id": item["item_id"], "scope_label": raw["scope"],
                         "confidence_1to3": raw["confidence"],
                         "notes": raw["basis"].replace("\n", " ")})
        toks_in += usage.get("prompt_tokens", 0)
        toks_out += usage.get("completion_tokens", 0)
        if i % 25 == 0 or i == len(items):
            print(f"    {model_key}: {i}/{len(items)} done, {len(failures)} failures",
                  flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["item_id", "scope_label",
                                                "confidence_1to3", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "annotator": model_key,
        "model_id": model_id,
        "n": len(items),
        "n_failed": len(failures),
        "prompt_tokens": toks_in,
        "completion_tokens": toks_out,
        "seconds": round(time.time() - start, 1),
        "output": str(out_path.relative_to(ROOT)),
        "failures": failures,
    }
    return summary


def agreement_report(paths: list[Path]) -> None:
    """Pairwise agreement plus agreement with the labels released in the paper."""
    mapping = {r["item_id"]: r for r in
               csv.DictReader((KIT / "item_id_map.csv").open(encoding="utf-8"))}
    labels: dict[str, dict[str, int]] = {}
    for path in paths:
        got = {}
        for row in csv.DictReader(path.open(encoding="utf-8")):
            raw = (row.get("scope_label") or "").strip()
            if raw in {"0", "1"}:
                got[row["item_id"]] = int(raw)
        labels[path.stem] = got

    names = list(labels)
    print("\n=== agreement with the released labels ===")
    for name in names:
        common = [i for i in labels[name] if i in mapping]
        agree = sum(1 for i in common
                    if labels[name][i] == int(mapping[i]["released_scope"]))
        print(f"  {name:34s} {agree:3d}/{len(common)}  {agree/len(common):.1%}")

    print("\n=== pairwise agreement ===")
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            common = sorted(set(labels[names[a]]) & set(labels[names[b]]))
            if not common:
                continue
            agree = sum(1 for i in common if labels[names[a]][i] == labels[names[b]][i])
            print(f"  {names[a]} vs {names[b]}: {agree}/{len(common)}  {agree/len(common):.1%}")

    # items where the LLM annotators disagree with each other, or where they
    # unanimously disagree with the released label: the adjudication shortlist
    shortlist = []
    for item in sorted(mapping):
        votes = [labels[n][item] for n in names if item in labels[n]]
        if len(votes) < 2:
            continue
        released = int(mapping[item]["released_scope"])
        if len(set(votes)) > 1 or (len(set(votes)) == 1 and votes[0] != released):
            shortlist.append((item, mapping[item]["record_id"], released, votes))
    print(f"\n=== adjudication shortlist: {len(shortlist)} items ===")
    for item, rid, released, votes in shortlist:
        print(f"  {item}  {rid:38s} released={released}  votes={votes}")


def adjudication_sheet(paths: list[Path], out_path: Path) -> None:
    """Write the disputed items for human adjudication.

    An item goes on the sheet when the LLM annotators disagree with each other,
    or when they agree with each other but differ from the label released in the
    paper. The human fills human_scope and human_note, and that column becomes
    the adjudicated label used by the scorer.
    """
    mapping = {r["item_id"]: r for r in
               csv.DictReader((KIT / "item_id_map.csv").open(encoding="utf-8"))}
    items = {r["item_id"]: r for r in
             csv.DictReader((KIT / "annotation_sheet.csv").open(encoding="utf-8"))}
    # real category per record, for the adjudicator's context
    manifest = json.loads((ROOT / "data" / "prompt_benchmark_v2.json").read_text())
    category = {r["record_id"]: r.get("category", "") for r in manifest["records"]}
    labels: dict[str, dict[str, str]] = {}
    bases: dict[str, dict[str, str]] = {}
    for path in paths:
        name = path.stem.replace("annotation_sheet_", "")
        labels[name] = {}
        bases[name] = {}
        for r in csv.DictReader(path.open(encoding="utf-8")):
            labels[name][r["item_id"]] = (r.get("scope_label") or "").strip()
            bases[name][r["item_id"]] = (r.get("notes") or "").replace("\n", " ")[:200]
    names = sorted(labels)

    rows = []
    for item in sorted(mapping):
        votes = [labels[n].get(item, "") for n in names]
        numeric = [int(v) for v in votes if v in {"0", "1"}]
        if len(numeric) < 2:
            continue
        released = int(mapping[item]["released_scope"])
        if len(set(numeric)) > 1 or (len(set(numeric)) == 1 and numeric[0] != released):
            rows.append({
                "item_id": item,
                "record_id": mapping[item]["record_id"],
                "category": category.get(mapping[item]["record_id"], ""),
                "prompt": items[item]["text"].replace("\n", " | "),
                "released_scope": released,
                **{n: labels[n].get(item, "") for n in names},
                **{f"{n}_basis": bases[n].get(item, "") for n in names},
                "human_scope": "",
                "human_note": "",
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["item_id", "record_id", "category", "prompt", "released_scope",
              *names, *[f"{n}_basis" for n in names], "human_scope", "human_note"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_path.relative_to(ROOT)} with {len(rows)} items to adjudicate")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=sorted(ANNOTATORS),
                    default=sorted(ANNOTATORS))
    ap.add_argument("--limit", type=int, default=None, help="label only the first N items")
    ap.add_argument("--report", action="store_true", help="only recompute agreement")
    ap.add_argument("--adjudication", action="store_true",
                    help="also write annotation_kit/adjudication_sheet.csv for human review")
    args = ap.parse_args()

    outdir = KIT / "llm_annotations"
    if args.report or args.adjudication:
        paths = sorted(outdir.glob("annotation_sheet_*.csv"))
        if not paths:
            raise SystemExit(f"no annotation files under {outdir}")
        agreement_report(paths)
        if args.adjudication:
            adjudication_sheet(paths, KIT / "adjudication_sheet.csv")
        return 0

    key = load_api_key()
    items = list(csv.DictReader((KIT / "annotation_sheet.csv").open(encoding="utf-8")))
    if args.limit:
        items = items[: args.limit]
    print(f"annotating {len(items)} items with {len(args.models)} model(s)")

    summaries = []
    for model_key in args.models:
        model_id = ANNOTATORS[model_key]
        print(f"\n[{model_key}] {model_id}", flush=True)
        out = outdir / f"annotation_sheet_{model_key}.csv"
        summary = annotate(model_key, model_id, key, items, out)
        summaries.append(summary)
        print(f"    -> {summary['output']}  failed={summary['n_failed']}  "
              f"in={summary['prompt_tokens']} out={summary['completion_tokens']} "
              f"({summary['seconds']}s)", flush=True)

    (outdir / "llm_annotation_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8")

    if not args.limit:
        paths = sorted(outdir.glob("annotation_sheet_*.csv"))
        agreement_report(paths)
        adjudication_sheet(paths, KIT / "adjudication_sheet.csv")
    print("\n__DONE__", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
