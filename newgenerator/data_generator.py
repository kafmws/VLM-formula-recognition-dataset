import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import math
import os
import random
import re
import urllib.error
import urllib.request
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

TOKEN_RE = re.compile(r"\\begin\{[^}]+\}|\\end\{[^}]+\}|\\[A-Za-z]+[*]?|\\[^A-Za-z\s]")
BEGIN_ENV_RE = re.compile(r"^\\begin\{([^}]+)\}$")
END_ENV_RE = re.compile(r"^\\end\{([^}]+)\}$")
ENV_TOKEN_RE = re.compile(r"\\(begin|end)\{([^}]+)\}")
BEGIN_ARRAY_RE = re.compile(r"\\begin\{array\}")
NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?![A-Za-z])")
VAR_RE = re.compile(r"(?<!\\)\b[a-zA-Z]\b")

DEFAULT_PROMPT = "<image>请根据图片中的公式生成对应的 latex 公式文本"

ARITY_2 = {r"\frac", r"\binom"}
ARITY_1 = {
    r"\sqrt",
    r"\mathbf",
    r"\mathbb",
    r"\mathcal",
    r"\mathrm",
    r"\text",
    r"\vec",
    r"\bar",
    r"\hat",
    r"\tilde",
    r"\overline",
    r"\boldsymbol",
    r"\mathfrak",
    r"\operatorname",
    r"\widehat",
    r"\pmb",
    r"\textbf",
}
IMMUTABLE_COMMANDS = {r"\begin", r"\end", r"\\", r"\,", r"\{", r"\}", r"\left", r"\right", r"\middle"}
NEVER_EMIT_COMMANDS = {r"\limits", r"\nolimits", r"\nonumber", r"\notag"}

GREEK_COMMANDS = {
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\delta",
    r"\epsilon",
    r"\varepsilon",
    r"\zeta",
    r"\eta",
    r"\theta",
    r"\vartheta",
    r"\iota",
    r"\kappa",
    r"\lambda",
    r"\mu",
    r"\nu",
    r"\xi",
    r"\pi",
    r"\rho",
    r"\sigma",
    r"\tau",
    r"\upsilon",
    r"\phi",
    r"\varphi",
    r"\chi",
    r"\psi",
    r"\omega",
    r"\Gamma",
    r"\Delta",
    r"\Theta",
    r"\Lambda",
    r"\Xi",
    r"\Pi",
    r"\Sigma",
    r"\Phi",
    r"\Psi",
    r"\Omega",
}
CALC_COMMANDS = {
    r"\int",
    r"\iint",
    r"\iiint",
    r"\oint",
    r"\sum",
    r"\prod",
    r"\lim",
    r"\limsup",
    r"\partial",
    r"\nabla",
    r"\infty",
}
FUNC_COMMANDS = {
    r"\sin",
    r"\cos",
    r"\tan",
    r"\cot",
    r"\sec",
    r"\csc",
    r"\sinh",
    r"\cosh",
    r"\tanh",
    r"\coth",
    r"\arcsin",
    r"\arctan",
    r"\exp",
    r"\log",
    r"\ln",
}
STYLE_COMMANDS = {
    r"\mathbf",
    r"\mathbb",
    r"\mathcal",
    r"\mathrm",
    r"\text",
    r"\textbf",
    r"\vec",
    r"\bar",
    r"\hat",
    r"\tilde",
    r"\overline",
    r"\boldsymbol",
    r"\mathfrak",
    r"\widehat",
}


@dataclass
class FormulaEntry:
    text: str
    parts: List[str]
    tokens: List[str]
    cmd_count: int
    length: int


class DynamicDistributor:
    def __init__(self, target_counter: Counter, alpha: float) -> None:
        cleaned = {k: int(v) for k, v in target_counter.items() if int(v) > 0}
        self.target = Counter(cleaned)
        self.current = Counter()
        self.alpha = alpha
        self.target_total = max(sum(self.target.values()), 1)
        self.vocab = max(len(self.target), 1)
        self.current_total = 0

    def observe(self, item: str) -> None:
        self.current[item] += 1
        self.current_total += 1

    def choose(self, candidates: List[str]) -> str:
        if not candidates:
            raise ValueError("candidates must not be empty")
        if len(candidates) == 1:
            return candidates[0]
        weights = [max(self._weight(c), 1e-9) for c in candidates]
        return random.choices(candidates, weights=weights, k=1)[0]

    def _weight(self, item: str) -> float:
        base = self.target.get(item, 1)
        target_p = base / self.target_total
        current_p = (self.current.get(item, 0) + 1) / (self.current_total + self.vocab)
        return base * ((target_p / current_p) ** self.alpha)


def parse_assistant_formula(item: Dict) -> str:
    messages = item.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return str(msg.get("content", "")).strip()
    return ""


def read_formulas(input_path: str) -> List[str]:
    formulas: List[str] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            text = parse_assistant_formula(item)
            if text:
                formulas.append(text)
    return formulas


def split_formula(text: str) -> Tuple[List[str], List[str]]:
    parts: List[str] = []
    tokens: List[str] = []
    last = 0
    for m in TOKEN_RE.finditer(text):
        parts.append(text[last : m.start()])
        tokens.append(m.group(0))
        last = m.end()
    parts.append(text[last:])
    return parts, tokens


def canonical_command(token: str) -> str:
    if BEGIN_ENV_RE.match(token):
        return r"\begin"
    if END_ENV_RE.match(token):
        return r"\end"
    return token


def arity_of(cmd: str) -> int:
    if cmd in ARITY_2:
        return 2
    if cmd in ARITY_1:
        return 1
    return 0


def command_family(cmd: str) -> str:
    if cmd in GREEK_COMMANDS:
        return "greek"
    if cmd in CALC_COMMANDS:
        return "calculus"
    if cmd in FUNC_COMMANDS:
        return "function"
    if cmd in STYLE_COMMANDS:
        return "style"
    return "other"


def build_entries(formulas: List[str]) -> Tuple[List[FormulaEntry], Counter]:
    entries: List[FormulaEntry] = []
    number_counter = Counter()
    for text in formulas:
        parts, tokens = split_formula(text)
        cmd_count = len(tokens)
        entries.append(FormulaEntry(text=text, parts=parts, tokens=tokens, cmd_count=cmd_count, length=len(text)))
        number_counter.update(NUMBER_RE.findall(text))
    return entries, number_counter


def parse_histogram(data: Dict) -> Counter:
    result = Counter()
    for k, v in data.items():
        try:
            key = int(k)
            val = int(v)
        except (TypeError, ValueError):
            continue
        if val > 0:
            result[key] = val
    return result


def load_stats(stats_path: str) -> Dict:
    if not os.path.exists(stats_path):
        return {}
    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "command_freq" in data:
        command_freq = Counter({k: int(v) for k, v in data.get("command_freq", {}).items() if int(v) > 0})
        env_freq = Counter({k: int(v) for k, v in data.get("env_freq", {}).items() if int(v) > 0})
        cmd_hist = parse_histogram(data.get("formula_command_count_hist", {}))
        len_hist = parse_histogram(data.get("formula_length_hist", {}))
        num_freq = Counter({k: int(v) for k, v in data.get("number_freq", {}).items() if int(v) > 0})
        return {
            "command_freq": command_freq,
            "env_freq": env_freq,
            "cmd_hist": cmd_hist,
            "len_hist": len_hist,
            "number_freq": num_freq,
        }

    command_freq = Counter()
    env_freq = Counter()
    for k, v in data.items():
        try:
            val = int(v)
        except (TypeError, ValueError):
            continue
        if val <= 0:
            continue
        if k.startswith("\\"):
            command_freq[k] = val
        else:
            env_freq[k] = val
    return {"command_freq": command_freq, "env_freq": env_freq, "cmd_hist": Counter(), "len_hist": Counter(), "number_freq": Counter()}


def sample_from_hist(hist: Counter, default_value: int) -> int:
    if not hist:
        return default_value
    keys = list(hist.keys())
    weights = list(hist.values())
    return random.choices(keys, weights=weights, k=1)[0]


def build_pools(command_freq: Counter) -> Tuple[Dict[Tuple[int, str], List[str]], Dict[int, List[str]]]:
    by_key: Dict[Tuple[int, str], List[str]] = defaultdict(list)
    by_arity: Dict[int, List[str]] = defaultdict(list)
    for cmd in command_freq:
        if cmd in IMMUTABLE_COMMANDS or cmd in NEVER_EMIT_COMMANDS:
            continue
        a = arity_of(cmd)
        fam = command_family(cmd)
        by_key[(a, fam)].append(cmd)
        by_arity[a].append(cmd)
    return by_key, by_arity


def candidate_commands(
    original_cmd: str, by_key: Dict[Tuple[int, str], List[str]], by_arity: Dict[int, List[str]]
) -> List[str]:
    if original_cmd in IMMUTABLE_COMMANDS:
        return [original_cmd]
    a = arity_of(original_cmd)
    fam = command_family(original_cmd)
    candidates = by_key.get((a, fam), []) or by_arity.get(a, [])
    candidates = [c for c in candidates if c not in NEVER_EMIT_COMMANDS]
    if not candidates:
        return [original_cmd]
    unique = list(dict.fromkeys(candidates + [original_cmd]))
    return unique


def braces_balanced(text: str) -> bool:
    bal = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text) and text[i + 1] in "{}":
            i += 2
            continue
        if ch == "{":
            bal += 1
        elif ch == "}":
            bal -= 1
            if bal < 0:
                return False
        i += 1
    return bal == 0


def env_pairs_balanced(text: str) -> bool:
    stack: List[str] = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", text):
        kind = m.group(1)
        env = m.group(2)
        if kind == "begin":
            stack.append(env)
        else:
            if not stack or stack[-1] != env:
                return False
            stack.pop()
    return not stack


def parse_brace_group(text: str, start: int) -> Tuple[int, int]:
    if start >= len(text) or text[start] != "{":
        return -1, -1
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text) and text[i + 1] in "{}":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, i + 1
            if depth < 0:
                return -1, -1
        i += 1
    return -1, -1


def find_matching_env_end(text: str, start_pos: int, env_name: str) -> int:
    depth = 1
    for m in ENV_TOKEN_RE.finditer(text, pos=start_pos):
        kind = m.group(1)
        env = m.group(2)
        if env != env_name:
            continue
        if kind == "begin":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return m.start()
    return -1


def infer_array_column_count(body: str) -> int:
    rows = re.split(r"\\\\(?:\[[^\]]*\])?", body)
    max_cols = 1
    for row in rows:
        stripped = row.strip()
        if not stripped:
            continue
        cols = stripped.count("&") + 1
        if cols > max_cols:
            max_cols = cols
    return max(1, min(max_cols, 12))


def normalize_array_colspec(text: str) -> str:
    replacements: List[Tuple[int, int, str]] = []
    for m in BEGIN_ARRAY_RE.finditer(text):
        after = m.end()
        while after < len(text) and text[after].isspace():
            after += 1
        group_start, group_end = parse_brace_group(text, after)
        body_start = group_end if group_start >= 0 else after
        end_pos = find_matching_env_end(text, m.end(), "array")
        if end_pos < 0:
            continue
        body = text[body_start:end_pos]
        col_count = infer_array_column_count(body)
        colspec = "{" + ("c" * col_count) + "}"
        if group_start >= 0:
            replacements.append((group_start, group_end, colspec))
        else:
            replacements.append((after, after, colspec))

    if not replacements:
        return text

    out = text
    for start, end, repl in reversed(replacements):
        out = out[:start] + repl + out[end:]
    return out


def arrays_have_colspec(text: str) -> bool:
    for m in BEGIN_ARRAY_RE.finditer(text):
        i = m.end()
        while i < len(text) and text[i].isspace():
            i += 1
        g0, g1 = parse_brace_group(text, i)
        if g0 < 0:
            return False
        content = text[g0 + 1 : g1 - 1].strip()
        if not content:
            return False
    return True


def is_formula_renderable(text: str) -> bool:
    if any(cmd in text for cmd in NEVER_EMIT_COMMANDS):
        return False
    if text.count(r"\left") != text.count(r"\right"):
        return False
    if not braces_balanced(text):
        return False
    if not env_pairs_balanced(text):
        return False
    if not arrays_have_colspec(text):
        return False
    return True


def make_img_path(img_dir: str, sample_index: int) -> str:
    if img_dir == "/dev/null":
        return "/dev/null"
    return os.path.join(img_dir, f"g_sample_{sample_index}.png")


def make_tmp_img_path(tmp_dir: str) -> str:
    return os.path.join(tmp_dir, f"tmp_{uuid.uuid4().hex}.png")


def render_formula_via_api(render_api_url: str, latex: str, img_path: str, timeout_sec: float) -> bool:
    payload = {"latex": latex, "img_path": img_path}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        render_api_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            if resp.status != 200:
                return False
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return False
    return bool(parsed.get("success"))

from baseline.infer_core.latex2img_file import LatexToImage
latexer = LatexToImage()
render_formula_via_api = latexer.latex_to_image

def generate_and_validate_candidate(
    entries: List[FormulaEntry],
    target_cmd_count: int,
    target_len: int,
    cmd_dist: DynamicDistributor,
    env_dist: DynamicDistributor,
    env_candidates: List[str],
    by_key: Dict[Tuple[int, str], List[str]],
    by_arity: Dict[int, List[str]],
    rewrite_prob: float,
    env_rewrite_prob: float,
    number_choices: List[str],
    number_mutation_prob: float,
    var_mutation_prob: float,
    render_api_url: str,
    render_timeout: float,
    precheck_render: bool,
) -> Tuple[bool, str, str, List[str], List[str]]:
    base = pick_base_entry(entries, target_cmd_count=target_cmd_count, target_len=target_len)
    formula, used_cmds, used_envs = mutate_entry(
        entry=base,
        cmd_dist=cmd_dist,
        env_dist=env_dist,
        env_candidates=env_candidates,
        by_key=by_key,
        by_arity=by_arity,
        rewrite_prob=rewrite_prob,
        env_rewrite_prob=env_rewrite_prob,
        number_choices=number_choices,
        number_mutation_prob=number_mutation_prob,
        var_mutation_prob=var_mutation_prob,
    )
    if not is_formula_renderable(formula):
        return False, "rule_check_failed", formula, used_cmds, used_envs

    if precheck_render:
        # Validate renderability without creating real image files.
        if not render_formula_via_api(
            # render_api_url=render_api_url,
            latex_string=formula,
            output_path="/dev/null",
            # timeout_sec=render_timeout,
        ):
            return False, "render_api_failed", formula, used_cmds, used_envs

    return True, "", formula, used_cmds, used_envs


def clone_distributor(source: DynamicDistributor) -> DynamicDistributor:
    cloned = DynamicDistributor(source.target, alpha=source.alpha)
    cloned.current = Counter(source.current)
    cloned.current_total = source.current_total
    return cloned


def generate_sample_slot(
    entries: List[FormulaEntry],
    target_cmd_count: int,
    target_len: int,
    cmd_dist: DynamicDistributor,
    env_dist: DynamicDistributor,
    env_candidates: List[str],
    by_key: Dict[Tuple[int, str], List[str]],
    by_arity: Dict[int, List[str]],
    rewrite_prob: float,
    env_rewrite_prob: float,
    number_choices: List[str],
    number_mutation_prob: float,
    var_mutation_prob: float,
    render_api_url: str,
    render_timeout: float,
    max_retries: int,
    precheck_render: bool,
    render_tmp_dir: str,
) -> Dict:
    local_cmd_dist = clone_distributor(cmd_dist)
    local_env_dist = clone_distributor(env_dist)
    attempts_used = 0
    last_reason = "unknown"
    last_formula = ""

    for _ in range(max(1, max_retries)):
        attempts_used += 1
        ok, reason, formula, used_cmds, used_envs = generate_and_validate_candidate(
            entries=entries,
            target_cmd_count=target_cmd_count,
            target_len=target_len,
            cmd_dist=local_cmd_dist,
            env_dist=local_env_dist,
            env_candidates=env_candidates,
            by_key=by_key,
            by_arity=by_arity,
            rewrite_prob=rewrite_prob,
            env_rewrite_prob=env_rewrite_prob,
            number_choices=number_choices,
            number_mutation_prob=number_mutation_prob,
            var_mutation_prob=var_mutation_prob,
            render_api_url=render_api_url,
            render_timeout=render_timeout,
            precheck_render=precheck_render,
        )
        if ok:
            rendered_path = ""
            if (not precheck_render) and render_tmp_dir:
                tmp_img_path = make_tmp_img_path(render_tmp_dir)
                if not render_formula_via_api(
                    latex_string=formula,
                    output_path=tmp_img_path,
                ):
                    if os.path.exists(tmp_img_path):
                        try:
                            os.remove(tmp_img_path)
                        except OSError:
                            pass
                    last_reason = "final_render_failed"
                    last_formula = formula
                    continue
                rendered_path = tmp_img_path
            return {
                "ok": True,
                "reason": "",
                "latex": formula,
                "used_cmds": used_cmds,
                "used_envs": used_envs,
                "attempts": attempts_used,
                "target_cmd_count": int(target_cmd_count),
                "target_len": int(target_len),
                "rendered_path": rendered_path,
            }
        last_reason = reason or "unknown"
        last_formula = formula

    return {
        "ok": False,
        "reason": last_reason,
        "latex": last_formula,
        "used_cmds": [],
        "used_envs": [],
        "attempts": attempts_used,
        "target_cmd_count": int(target_cmd_count),
        "target_len": int(target_len),
        "rendered_path": "",
    }


def open_progress_bar(total: int):
    if tqdm is None:
        return None
    return tqdm(total=total, desc="Generating", unit="sample")


def pick_base_entry(entries: List[FormulaEntry], target_cmd_count: int, target_len: int, sample_k: int = 128) -> FormulaEntry:
    if len(entries) <= sample_k:
        candidates = entries
    else:
        candidates = random.sample(entries, sample_k)
    ranked = sorted(
        candidates,
        key=lambda e: 3 * abs(e.cmd_count - target_cmd_count) + abs(e.length - target_len),
    )
    top_n = min(8, len(ranked))
    return random.choice(ranked[:top_n])


def mutate_numbers(text: str, number_choices: List[str], mutation_prob: float) -> str:
    if not number_choices:
        return text

    def repl(match: re.Match) -> str:
        if random.random() < mutation_prob:
            return random.choice(number_choices)
        return match.group(0)

    return NUMBER_RE.sub(repl, text)


def mutate_variables(text: str, mutation_prob: float) -> str:
    pool = ["x", "y", "z", "t", "u", "v", "i", "j", "k", "m", "n", "p", "q"]

    def repl(match: re.Match) -> str:
        if random.random() < mutation_prob:
            return random.choice(pool)
        return match.group(0)

    return VAR_RE.sub(repl, text)


def mutate_entry(
    entry: FormulaEntry,
    cmd_dist: DynamicDistributor,
    env_dist: DynamicDistributor,
    env_candidates: List[str],
    by_key: Dict[Tuple[int, str], List[str]],
    by_arity: Dict[int, List[str]],
    rewrite_prob: float,
    env_rewrite_prob: float,
    number_choices: List[str],
    number_mutation_prob: float,
    var_mutation_prob: float,
) -> Tuple[str, List[str], List[str]]:
    env_map: Dict[str, str] = {}
    new_tokens: List[str] = []
    used_commands: List[str] = []
    used_envs: List[str] = []

    for token in entry.tokens:
        begin_m = BEGIN_ENV_RE.match(token)
        end_m = END_ENV_RE.match(token)

        if begin_m:
            old_env = begin_m.group(1)
            if old_env not in env_map:
                if env_candidates and random.random() < env_rewrite_prob:
                    env_map[old_env] = env_dist.choose(env_candidates)
                else:
                    env_map[old_env] = old_env
            new_env = env_map[old_env]
            new_tokens.append(rf"\begin{{{new_env}}}")
            used_commands.append(r"\begin")
            used_envs.append(new_env)
            continue

        if end_m:
            old_env = end_m.group(1)
            if old_env not in env_map:
                env_map[old_env] = old_env
            new_env = env_map[old_env]
            new_tokens.append(rf"\end{{{new_env}}}")
            used_commands.append(r"\end")
            used_envs.append(new_env)
            continue

        cmd = canonical_command(token)
        chosen = cmd
        if cmd not in IMMUTABLE_COMMANDS and random.random() < rewrite_prob:
            candidates = candidate_commands(cmd, by_key, by_arity)
            chosen = cmd_dist.choose(candidates)
        if chosen in NEVER_EMIT_COMMANDS:
            new_tokens.append("")
            continue
        new_tokens.append(chosen)
        used_commands.append(chosen)

    combined: List[str] = []
    for i, token in enumerate(new_tokens):
        combined.append(entry.parts[i])
        combined.append(token)
    combined.append(entry.parts[-1])
    text = "".join(combined)

    text = mutate_numbers(text, number_choices=number_choices, mutation_prob=number_mutation_prob)
    text = mutate_variables(text, mutation_prob=var_mutation_prob)
    text = normalize_array_colspec(text)
    return text, used_commands, used_envs


def commands_and_envs_from_entry(entry: FormulaEntry) -> Tuple[List[str], List[str]]:
    commands: List[str] = []
    envs: List[str] = []
    for token in entry.tokens:
        begin_m = BEGIN_ENV_RE.match(token)
        end_m = END_ENV_RE.match(token)
        if begin_m:
            commands.append(r"\begin")
            envs.append(begin_m.group(1))
            continue
        if end_m:
            commands.append(r"\end")
            envs.append(end_m.group(1))
            continue
        cmd = canonical_command(token)
        if cmd in NEVER_EMIT_COMMANDS:
            continue
        commands.append(cmd)
    return commands, envs


def l1_distance(target: Counter, current: Counter) -> float:
    target_total = max(sum(target.values()), 1)
    current_total = max(sum(current.values()), 1)
    keys = set(target.keys()) | set(current.keys())
    return sum(abs(target.get(k, 0) / target_total - current.get(k, 0) / current_total) for k in keys)


def js_divergence(target: Counter, current: Counter) -> float:
    target_total = max(sum(target.values()), 1)
    current_total = max(sum(current.values()), 1)
    keys = set(target.keys()) | set(current.keys())
    divergence = 0.0
    for k in keys:
        p = target.get(k, 0) / target_total
        q = current.get(k, 0) / current_total
        m = 0.5 * (p + q)
        if p > 0:
            divergence += 0.5 * p * math.log(p / m, 2)
        if q > 0:
            divergence += 0.5 * q * math.log(q / m, 2)
    return divergence


def print_distribution_report(name: str, target: Counter, current: Counter, topk: int) -> Tuple[float, float]:
    target_total = max(sum(target.values()), 1)
    current_total = max(sum(current.values()), 1)
    l1 = l1_distance(target, current)
    js = js_divergence(target, current)
    print(f"\n===== {name} Distribution =====")
    print(f"L1 distance: {l1:.6f}")
    print(f"JS divergence: {js:.6f}")
    for cmd, cnt in target.most_common(topk):
        t = cnt / target_total
        c = current.get(cmd, 0) / current_total
        print(f"{cmd:16s} target={t:.4f} current={c:.4f} diff={abs(t - c):.4f}")
    return l1, js


def write_output(samples: List[str], output_path: str, output_format: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for formula in samples:
            if output_format == "chat":
                item = {
                    "messages": [
                        {"role": "user", "content": DEFAULT_PROMPT},
                        {"role": "assistant", "content": formula},
                    ]
                }
            else:
                item = formula
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate augmented LaTeX samples with matched command distribution.")
    parser.add_argument("--input", default="./train_mini.jsonl", help="Input training JSONL.")
    parser.add_argument("--stats", default="./orignal.json", help="Stats JSON generated by token_stat.py.")
    parser.add_argument("--output", default="./generated_latex.jsonl", help="Output JSONL path.")
    parser.add_argument("--num-samples", type=int, default=3000, help="Number of generated samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--rewrite-prob", type=float, default=0.45, help="Command replacement probability per token.")
    parser.add_argument("--env-rewrite-prob", type=float, default=0.55, help="Environment replacement probability.")
    parser.add_argument("--number-mutation-prob", type=float, default=0.20, help="Number mutation probability.")
    parser.add_argument("--var-mutation-prob", type=float, default=0.12, help="Variable mutation probability.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Dynamic distribution correction strength.")
    parser.add_argument("--max-retries", type=int, default=8, help="Max retries when generated formula fails renderability checks.")
    parser.add_argument("--render-api-url", default="http://127.0.0.1:8000/render", help="LaTeX render API URL.")
    parser.add_argument("--render-timeout", type=float, default=10.0, help="Timeout (seconds) for each render API call.")
    parser.add_argument("--img-dir", default="/dev/null", help="Image output dir. Use /dev/null for render-only validation.")
    parser.add_argument("--error-output", default="./error.jsonl", help="Failed sample records JSONL path.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for independent sample slots.")
    parser.add_argument(
        "--max-total-attempts",
        type=int,
        default=0,
        help="Global cap for candidate attempts. <=0 means auto: num_samples * max_retries * 4.",
    )
    parser.add_argument("--format", choices=["latex", "chat"], default="latex", help="Output JSONL format.")
    parser.add_argument("--topk", "--top", dest="topk", type=int, default=20, help="Top-K report entries.")
    parser.add_argument("--strict-verify", action="store_true", help="Fail with non-zero exit code if distribution is out of thresholds.")
    parser.add_argument("--max-cmd-l1", type=float, default=0.10, help="Max allowed command L1 distance for strict verification.")
    parser.add_argument("--max-env-l1", type=float, default=0.15, help="Max allowed env L1 distance for strict verification.")
    parser.add_argument("--max-cmd-js", type=float, default=0.03, help="Max allowed command JS divergence for strict verification.")
    parser.add_argument("--max-env-js", type=float, default=0.05, help="Max allowed env JS divergence for strict verification.")
    args = parser.parse_args()

    random.seed(args.seed)
    if args.img_dir != "/dev/null":
        os.makedirs(args.img_dir, exist_ok=True)
    render_tmp_dir = ""
    if args.img_dir != "/dev/null":
        render_tmp_dir = os.path.join(args.img_dir, ".tmp_render")
        os.makedirs(render_tmp_dir, exist_ok=True)
    error_dir = os.path.dirname(os.path.abspath(args.error_output))
    if error_dir:
        os.makedirs(error_dir, exist_ok=True)

    formulas = read_formulas(args.input)
    if not formulas:
        raise ValueError(f"No formulas parsed from {args.input}")
    entries, source_number_counter = build_entries(formulas)
    valid_entries = [e for e in entries if is_formula_renderable(e.text)]
    if valid_entries:
        entries = valid_entries
    if not entries:
        raise ValueError("No renderable base formulas after validation.")

    stats = load_stats(args.stats)
    target_cmd = stats.get("command_freq", Counter())
    target_env = stats.get("env_freq", Counter())
    cmd_hist = stats.get("cmd_hist", Counter())
    len_hist = stats.get("len_hist", Counter())
    num_freq = stats.get("number_freq", Counter())

    if not target_cmd:
        local_cmd = Counter()
        for e in entries:
            for tok in e.tokens:
                local_cmd[canonical_command(tok)] += 1
        target_cmd = local_cmd

    if not target_env:
        local_env = Counter()
        for e in entries:
            for tok in e.tokens:
                m = BEGIN_ENV_RE.match(tok)
                if m:
                    local_env[m.group(1)] += 1
        target_env = local_env

    if not cmd_hist:
        cmd_hist = Counter(e.cmd_count for e in entries)
    if not len_hist:
        len_hist = Counter(e.length for e in entries)
    if not num_freq:
        num_freq = source_number_counter

    by_key, by_arity = build_pools(target_cmd)
    number_choices = [k for k, _ in num_freq.most_common()]
    env_candidates = list(target_env.keys())

    cmd_dist = DynamicDistributor(target_cmd, alpha=args.alpha)
    env_dist = DynamicDistributor(target_env if target_env else Counter({"bmatrix": 1}), alpha=args.alpha)

    mean_cmd = round(sum(k * v for k, v in cmd_hist.items()) / max(sum(cmd_hist.values()), 1))
    mean_len = round(sum(k * v for k, v in len_hist.items()) / max(sum(len_hist.values()), 1))

    samples: List[str] = []
    total_attempts = 0
    failed_slots = 0
    error_records = 0
    workers = max(1, int(args.workers))
    max_total_attempts = (
        args.max_total_attempts
        if args.max_total_attempts > 0
        else max(args.num_samples * max(args.max_retries, 1) * 4, args.num_samples)
    )
    pbar = open_progress_bar(args.num_samples)
    precheck_render = args.img_dir == "/dev/null"
    with open(args.error_output, "w", encoding="utf-8") as err_f, ThreadPoolExecutor(max_workers=workers) as executor:
        inflight: Dict = {}
        pending_budget = 0

        while inflight or (len(samples) < args.num_samples and total_attempts + pending_budget < max_total_attempts):
            while (
                len(inflight) < workers
                and len(samples) + len(inflight) < args.num_samples
                and total_attempts + pending_budget < max_total_attempts
            ):
                budget_left = max_total_attempts - total_attempts - pending_budget
                slot_retry_budget = min(args.max_retries, budget_left)
                if slot_retry_budget <= 0:
                    break

                target_cmd_count = sample_from_hist(cmd_hist, default_value=mean_cmd)
                target_len = sample_from_hist(len_hist, default_value=mean_len)
                cmd_snapshot = clone_distributor(cmd_dist)
                env_snapshot = clone_distributor(env_dist)
                fut = executor.submit(
                    generate_sample_slot,
                    entries=entries,
                    target_cmd_count=target_cmd_count,
                    target_len=target_len,
                    cmd_dist=cmd_snapshot,
                    env_dist=env_snapshot,
                    env_candidates=env_candidates,
                    by_key=by_key,
                    by_arity=by_arity,
                    rewrite_prob=args.rewrite_prob,
                    env_rewrite_prob=args.env_rewrite_prob,
                    number_choices=number_choices,
                    number_mutation_prob=args.number_mutation_prob,
                    var_mutation_prob=args.var_mutation_prob,
                    render_api_url=args.render_api_url,
                    render_timeout=args.render_timeout,
                    max_retries=slot_retry_budget,
                    precheck_render=precheck_render,
                    render_tmp_dir=render_tmp_dir,
                )
                inflight[fut] = slot_retry_budget
                pending_budget += slot_retry_budget

            if not inflight:
                break

            done, _ = wait(list(inflight.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                reserved = int(inflight.pop(future, 0))
                pending_budget = max(0, pending_budget - reserved)
                result = future.result()
                total_attempts += int(result.get("attempts", 0))
                target_cmd_count = int(result.get("target_cmd_count", mean_cmd))
                target_len = int(result.get("target_len", mean_len))

                if not result.get("ok"):
                    failed_slots += 1
                    error_item = {
                        "target_index": len(samples),
                        "target_cmd_count": target_cmd_count,
                        "target_len": target_len,
                        "reason": result.get("reason", "unknown"),
                        "latex": result.get("latex", ""),
                    }
                    err_f.write(json.dumps(error_item, ensure_ascii=False) + "\n")
                    error_records += 1
                    if pbar is not None:
                        pbar.set_postfix(
                            attempts=total_attempts,
                            failed=failed_slots,
                            errors=error_records,
                        )
                    continue

                accepted_formula = str(result.get("latex", ""))
                accepted_cmds = list(result.get("used_cmds", []))
                accepted_envs = list(result.get("used_envs", []))
                rendered_path = str(result.get("rendered_path", ""))

                if len(samples) >= args.num_samples:
                    if rendered_path and os.path.exists(rendered_path):
                        try:
                            os.remove(rendered_path)
                        except OSError:
                            pass
                    continue

                if args.img_dir != "/dev/null":
                    if not rendered_path or not os.path.exists(rendered_path):
                        failed_slots += 1
                        error_item = {
                            "target_index": len(samples),
                            "target_cmd_count": target_cmd_count,
                            "target_len": target_len,
                            "reason": "rendered_image_missing",
                            "latex": accepted_formula,
                        }
                        err_f.write(json.dumps(error_item, ensure_ascii=False) + "\n")
                        error_records += 1
                        if pbar is not None:
                            pbar.set_postfix(
                                attempts=total_attempts,
                                failed=failed_slots,
                                errors=error_records,
                            )
                        continue

                    img_path = make_img_path(args.img_dir, len(samples))
                    try:
                        os.replace(rendered_path, img_path)
                    except OSError:
                        failed_slots += 1
                        error_item = {
                            "target_index": len(samples),
                            "target_cmd_count": target_cmd_count,
                            "target_len": target_len,
                            "reason": "final_image_move_failed",
                            "latex": accepted_formula,
                        }
                        err_f.write(json.dumps(error_item, ensure_ascii=False) + "\n")
                        error_records += 1
                        if pbar is not None:
                            pbar.set_postfix(
                                attempts=total_attempts,
                                failed=failed_slots,
                                errors=error_records,
                            )
                        continue

                for cmd in accepted_cmds:
                    cmd_dist.observe(cmd)
                for env in accepted_envs:
                    env_dist.observe(env)

                samples.append(accepted_formula)
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        attempts=total_attempts,
                        failed=failed_slots,
                        errors=error_records,
                    )
    if pbar is not None:
        pbar.close()

    write_output(samples, args.output, args.format)
    print(f"Generated samples: {len(samples)}")
    print(f"Total candidate attempts: {total_attempts}")
    print(f"Failed slots: {failed_slots}")
    print(f"Error records saved: {error_records}")
    print(f"Error file: {os.path.abspath(args.error_output)}")
    if len(samples) < args.num_samples:
        print(
            f"Warning: only generated {len(samples)} successful samples "
            f"(target={args.num_samples}, max_total_attempts={max_total_attempts})."
        )
    print(f"Saved to: {os.path.abspath(args.output)}")

    cmd_l1, cmd_js = print_distribution_report("Command", target_cmd, cmd_dist.current, args.topk)
    env_l1, env_js = 0.0, 0.0
    if target_env:
        env_l1, env_js = print_distribution_report("Environment", target_env, env_dist.current, min(args.topk, len(target_env)))
    avg_len = sum(len(s) for s in samples) / max(len(samples), 1)
    print(f"\nGenerated avg length: {avg_len:.2f}")

    if args.strict_verify:
        errors = []
        if cmd_l1 > args.max_cmd_l1:
            errors.append(f"command L1 {cmd_l1:.6f} > {args.max_cmd_l1}")
        if cmd_js > args.max_cmd_js:
            errors.append(f"command JS {cmd_js:.6f} > {args.max_cmd_js}")
        if target_env:
            if env_l1 > args.max_env_l1:
                errors.append(f"env L1 {env_l1:.6f} > {args.max_env_l1}")
            if env_js > args.max_env_js:
                errors.append(f"env JS {env_js:.6f} > {args.max_env_js}")
        if errors:
            print("\nSTRICT VERIFY: FAILED")
            for e in errors:
                print(f"- {e}")
            raise SystemExit(2)
        print("\nSTRICT VERIFY: PASSED")


if __name__ == "__main__":
    main()
