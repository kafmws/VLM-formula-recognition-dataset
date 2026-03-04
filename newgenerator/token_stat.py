import argparse
import json
import math
import os
import re
from collections import Counter
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

TOKEN_RE = re.compile(
    r"""
    \\begin\{(?P<begin>[^}]+)\}
    |\\end\{(?P<end>[^}]+)\}
    |\\(?P<word>[A-Za-z]+[*]?)
    |\\(?P<symbol>[^A-Za-z\s])
    """,
    re.VERBOSE,
)
NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?![A-Za-z])")
BIGRAM_SEP = "|||"


def quantile(values: List[int], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    pos = q * (len(sorted_values) - 1)
    left = math.floor(pos)
    right = math.ceil(pos)
    if left == right:
        return float(sorted_values[left])
    weight = pos - left
    return sorted_values[left] * (1.0 - weight) + sorted_values[right] * weight


def parse_assistant_formula(item: Dict) -> str:
    messages = item.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role", "") == "assistant":
            return str(msg.get("content", "")).strip()
    return ""


def iter_formulas(jsonl_path: str) -> Iterable[str]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            text = parse_assistant_formula(item) if isinstance(item, Dict) else item
            text = re.sub(r"\n", " ", text.strip())
            text = re.sub(r"\s+", " ", text.strip())
            if text:
                yield text


def extract_tokens(text: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    commands: List[str] = []
    envs: List[str] = []
    symbols: List[str] = []
    numbers = NUMBER_RE.findall(text)
    for m in TOKEN_RE.finditer(text):
        begin_env = m.group("begin")
        end_env = m.group("end")
        word = m.group("word")
        symbol = m.group("symbol")
        if begin_env is not None:
            commands.append(r"\begin")
            envs.append(begin_env)
        elif end_env is not None:
            commands.append(r"\end")
            envs.append(end_env)
        elif word is not None:
            commands.append("\\" + word)
        elif symbol is not None:
            token = "\\" + symbol
            commands.append(token)
            symbols.append(token)
    return commands, envs, symbols, numbers


def counter_to_sorted_dict(counter: Counter) -> Dict[str, int]:
    return {k: int(v) for k, v in counter.most_common()}


def histogram_to_dict(values: List[int]) -> Dict[str, int]:
    return {str(k): int(v) for k, v in Counter(values).most_common()}


def build_report(input_path: str, topk: int) -> Dict:
    cmd_counter = Counter()
    env_counter = Counter()
    symbol_counter = Counter()
    number_counter = Counter()
    bigram_counter = Counter()
    formula_lengths: List[int] = []
    formula_cmd_counts: List[int] = []
    formula_env_counts: List[int] = []

    for text in iter_formulas(input_path):
        commands, envs, symbols, numbers = extract_tokens(text)
        cmd_counter.update(commands)
        env_counter.update(envs)
        symbol_counter.update(symbols)
        number_counter.update(numbers)
        formula_lengths.append(len(text))
        formula_cmd_counts.append(len(commands))
        formula_env_counts.append(len(envs))

        seq = ["<BOS>", *commands, "<EOS>"]
        for left, right in zip(seq, seq[1:]):
            bigram_counter[f"{left}{BIGRAM_SEP}{right}"] += 1

    sample_count = len(formula_lengths)
    avg_len = (sum(formula_lengths) / sample_count) if sample_count else 0.0
    avg_cmd = (sum(formula_cmd_counts) / sample_count) if sample_count else 0.0

    report = {
        "meta": {
            "input_file": os.path.abspath(input_path),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "num_samples": sample_count,
        },
        "summary": {
            "avg_formula_length": avg_len,
            "avg_command_count": avg_cmd,
            "num_unique_commands": len(cmd_counter),
            "num_unique_envs": len(env_counter),
            "formula_length_quantiles": {
                "p50": quantile(formula_lengths, 0.50),
                "p90": quantile(formula_lengths, 0.90),
                "p95": quantile(formula_lengths, 0.95),
                "max": max(formula_lengths) if formula_lengths else 0,
            },
            "command_count_quantiles": {
                "p50": quantile(formula_cmd_counts, 0.50),
                "p90": quantile(formula_cmd_counts, 0.90),
                "p95": quantile(formula_cmd_counts, 0.95),
                "max": max(formula_cmd_counts) if formula_cmd_counts else 0,
            },
        },
        "command_freq": counter_to_sorted_dict(cmd_counter),
        "command_prob": {
            k: (v / max(sum(cmd_counter.values()), 1))
            for k, v in counter_to_sorted_dict(cmd_counter).items()
        },
        "single_symbol_freq": counter_to_sorted_dict(symbol_counter),
        "env_freq": counter_to_sorted_dict(env_counter),
        "number_freq": counter_to_sorted_dict(number_counter),
        "command_bigram_freq": counter_to_sorted_dict(bigram_counter),
        "formula_length_hist": histogram_to_dict(formula_lengths),
        "formula_command_count_hist": histogram_to_dict(formula_cmd_counts),
        "formula_env_count_hist": histogram_to_dict(formula_env_counts),
        "top_commands": [[k, int(v)] for k, v in cmd_counter.most_common(topk)],
        "top_envs": [[k, int(v)] for k, v in env_counter.most_common(topk)],
    }
    return report


def print_summary(report: Dict, topk: int) -> None:
    summary = report["summary"]
    meta = report["meta"]
    print("===== Dataset Summary =====")
    print(f"Samples: {meta['num_samples']}")
    print(f"Avg formula length: {summary['avg_formula_length']:.2f}")
    print(f"Avg command count: {summary['avg_command_count']:.2f}")
    print(f"Unique commands: {summary['num_unique_commands']}")
    print(f"Unique envs: {summary['num_unique_envs']}")

    print(f"\n===== Top {topk} Commands =====")
    for cmd, count in report["top_commands"]:
        print(f"{cmd:16s} {count}")

    print(f"\n===== Top {topk} Envs =====")
    for env, count in report["top_envs"]:
        print(f"{env:16s} {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect LaTeX command distribution from JSONL dataset.")
    parser.add_argument("--input", default="./train_mini.jsonl", help="Input JSONL file path.")
    parser.add_argument("--output", default="./orignal.json", help="Output JSON report path.")
    parser.add_argument("--topk", "--top", dest="topk", type=int, default=20, help="Top-K items to print in CLI.")
    args = parser.parse_args()

    report = build_report(args.input, args.topk)
    print_summary(report, args.topk)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved report: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
