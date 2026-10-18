"""Compare models/prompts at rewriting a weak CV bullet into an impact statement.

    python -m evals.run gpt-4o-mini@baseline gpt-4o-mini@tuned ollama/qwen2.5:3b@tuned [--limit N]

A candidate is `model@prompt`. A `ollama/` model prefix talks to a local Ollama server.
Cheap deterministic checks run first; an LLM judge then scores overall quality 1-5.
"""
import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langfuse.openai import OpenAI  # noqa: E402  (traced like the app's calls)

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
WEAK_STARTS = {"responsible", "worked", "helped", "involved", "tasked", "duties", "i", "participated"}

PROMPTS = {
    "baseline": "Rewrite this CV bullet to be stronger. Reply with the bullet only.",
    "tuned": """You rewrite CV bullets into impact statements.
Rules:
- One line, at most 25 words, starting with a strong past-tense action verb.
- Format: action + what + result/impact.
- Keep every fact from the original. NEVER invent numbers, tools or outcomes.
- If the impact is unknown, add a placeholder such as [X%] or [metric] instead of guessing.
- Reply with the rewritten bullet only.""",
}

JUDGE = """Score a rewritten CV bullet from 1 to 5. Be strict about honesty.
Invented content = any outcome, result, metric, tool, scope or claim NOT stated in the original
(e.g. "increased traffic", "exceptional", "cross-functional" when the original says nothing of the kind).
Placeholders like [X%] or [metric] are NOT invented content; they are the honest way to mark unknown impact.
Scoring:
- Any invented content: maximum 2.
- No invented content: 3 = plain but accurate; 4 = strong action verb + concise; 5 = also has a clear impact slot (real or placeholder).
Reply as JSON: {{"score": <1-5>, "reason": "<short>"}}

Original: {original}
Rewrite: {rewrite}"""


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", text))


def check(original: str, rewrite: str) -> dict[str, bool]:
    unbracketed = re.sub(r"\[[^\]]*\]", "", rewrite)  # placeholders like [X%] are allowed
    words = rewrite.split()
    return {
        "no_invented_numbers": numbers(unbracketed) <= numbers(original),
        "concise": 0 < len(words) <= 30,
        "strong_start": bool(words) and words[0].lower().strip("-•*") not in WEAK_STARTS,
        "single_line": "\n" not in rewrite.strip(),
    }


def make_client(model: str) -> tuple[OpenAI, str]:
    if model.startswith("ollama/"):
        return OpenAI(base_url=OLLAMA_URL, api_key="ollama"), model.removeprefix("ollama/")
    return OpenAI(), model


def rewrite(candidate: str, bullet: str) -> tuple[str, float, int]:
    model, prompt = candidate.split("@")
    client, name = make_client(model)
    start = time.perf_counter()
    res = client.chat.completions.create(
        model=name, temperature=0.3,
        messages=[{"role": "system", "content": PROMPTS[prompt]}, {"role": "user", "content": bullet}],
    )
    text = res.choices[0].message.content.strip().lstrip("-•* ").strip()
    return text, time.perf_counter() - start, res.usage.total_tokens if res.usage else 0


def judge(original: str, rewritten: str) -> dict:
    res = OpenAI().chat.completions.create(
        model=JUDGE_MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": JUDGE.format(original=original, rewrite=rewritten)}],
    )
    return json.loads(res.choices[0].message.content)


def run(candidates: list[str], limit: int | None) -> None:
    rows = [json.loads(l) for l in Path("evals/dataset.jsonl").read_text().splitlines()][:limit]
    # ponytail: sequential calls, fine for ~40 rows x few candidates; add a thread pool if it gets slow
    summary, details = [], {}
    for cand in candidates:
        results = []
        for row in rows:
            text, secs, tokens = rewrite(cand, row["bullet"])
            checks, verdict = check(row["bullet"], text), judge(row["bullet"], text)
            results.append({"bullet": row["bullet"], "rewrite": text, "checks": checks,
                            "score": verdict["score"], "reason": verdict["reason"], "secs": secs, "tokens": tokens})
        details[cand] = results
        n = len(results)
        summary.append({
            "candidate": cand,
            "checks_pass": sum(all(r["checks"].values()) for r in results) / n,
            "no_invented": sum(r["checks"]["no_invented_numbers"] for r in results) / n,
            "judge": statistics.mean(r["score"] for r in results),
            "secs": statistics.mean(r["secs"] for r in results),
            "tokens": statistics.mean(r["tokens"] for r in results),
        })
    print(f"\n{'candidate':34}{'checks':>8}{'no-invent':>11}{'judge/5':>9}{'sec':>7}{'tokens':>8}")
    for s in summary:
        print(f"{s['candidate']:34}{s['checks_pass']:>8.0%}{s['no_invented']:>11.0%}{s['judge']:>9.2f}{s['secs']:>7.2f}{s['tokens']:>8.0f}")
    out = Path("evals/results") / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"judge_model": JUDGE_MODEL, "summary": summary, "details": details}, indent=2))
    print(f"\nPer-example results: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", nargs="+")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    run(args.candidates, args.limit)
