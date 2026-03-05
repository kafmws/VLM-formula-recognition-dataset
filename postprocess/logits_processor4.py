from posixpath import curdir
import re
import os
import torch
from transformers import LogitsProcessor, AutoTokenizer, Qwen2TokenizerFast, AutoModel

import torch
import re
from transformers import LogitsProcessor, LogitsProcessorList
from dataclasses import dataclass, field
from typing import Optional


# ================================================================
# LaTeX 合法命令白名单
# ================================================================

VALID_COMMANDS = {
    'frac', 'sqrt', 'sum', 'prod', 'int', 'oint', 'lim', 'infty',
    'partial', 'nabla', 'cdot', 'cdots', 'ldots', 'vdots', 'ddots',
    'left', 'right', 'big', 'Big', 'bigg', 'Bigg',
    'langle', 'rangle', 'lfloor', 'rfloor', 'lceil', 'rceil',
    'lvert', 'rvert', 'lVert', 'rVert', 'vert', 'Vert',
    'begin', 'end',
    'matrix', 'pmatrix', 'bmatrix', 'vmatrix', 'Vmatrix', 'cases',
    'aligned', 'align', 'equation', 'gather', 'array', 'smallmatrix',
    'eqnarray', 'subarray',
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'varepsilon',
    'zeta', 'eta', 'theta', 'vartheta', 'iota', 'kappa', 'lambda',
    'mu', 'nu', 'xi', 'pi', 'varpi', 'rho', 'varrho', 'sigma', 'varkappa',
    'varsigma', 'tau', 'upsilon', 'phi', 'varphi', 'chi', 'psi', 'omega',
    'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma',
    'Upsilon', 'Phi', 'Psi', 'Omega',
    'mathbb', 'mathbf', 'mathcal', 'mathfrak', 'mathit', 'mathrm', 'mathsf',
    'boldsymbol', 'text', 'mbox', 'operatorname',
    'times', 'div', 'pm', 'mp', 'oplus', 'otimes', 'circ', 'bullet',
    'cap', 'cup', 'vee', 'wedge', 'setminus',
    'leq', 'geq', 'neq', 'approx', 'equiv', 'sim', 'simeq', 'cong',
    'subset', 'supset', 'subseteq', 'supseteq', 'in', 'notin',
    'll', 'gg', 'prec', 'succ',
    'to', 'rightarrow', 'leftarrow', 'Rightarrow', 'Leftarrow',
    'leftrightarrow', 'Leftrightarrow', 'mapsto',
    'uparrow', 'downarrow', 'updownarrow', 'Uparrow', 'Downarrow',
    'nearrow', 'searrow', 'swarrow', 'nwarrow',
    'xrightarrow', 'xleftarrow',
    'forall', 'exists', 'neg', 'top', 'bot', 'emptyset', 'varnothing',
    'overline', 'underline', 'hat', 'tilde', 'bar', 'vec', 'dot', 'ddot',
    'widehat', 'widetilde', 'overrightarrow', 'overleftarrow',
    'overbrace', 'underbrace', 'overset', 'underset', 'stackrel',
    'limits', 'nolimits', 'displaystyle', 'textstyle', 'scriptstyle',
    'quad', 'qquad', 'hspace', 'vspace',
    'log', 'ln', 'exp', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
    'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh',
    'max', 'min', 'sup', 'inf', 'det', 'dim', 'ker', 'deg',
    'gcd', 'mod', 'bmod', 'pmod', 'lcm',
    'Re', 'Im', 'arg', 'sgn',
    'binom', 'choose', 'over', 'atop',
    'hline', 'cline', 'multicolumn', 'multirow',
    'color', 'textcolor',
    'not', 'nonumber', 'label', 'tag',
    'perp', 'parallel', 'angle', 'measuredangle',
    'therefore', 'because', 'propto',
    'substack', 'sideset',
    'gets', 'iff', 'implies',
    'complement', 'triangle', 'square',
    'DeclareMathOperator',
}

MATRIX_ENVS = {
    "matrix", "pmatrix", "bmatrix", "vmatrix", "Vmatrix",
    "smallmatrix", "array", "aligned", "align", "cases",
    "gather", "eqnarray", "subarray",
}

BRACKET_PAIRS = {')': '(', ']': '[', '}': '{'}
OPEN_BRACKETS = set('([{')
CLOSE_BRACKETS = set(')]}')
# 括号不匹配时禁止的闭合符号
BRACKET_MISMATCH = {
    '(': (']', '}'),
    '[': (')', '}'),
    '{': (')', ']'),
}


# ================================================================
# LatexState：追踪单个样本的生成状态
# ================================================================

@dataclass
class LatexState:
    """追踪单个样本已生成序列的语法状态"""

    generated_text: str = ""
    bracket_stack: list = field(default_factory=list)
    env_stack: list = field(default_factory=list)
    in_matrix_env: bool = False

    # 上标/下标状态
    last_script_char: str = ""   # 最近一次 ^ 或 _ 的符号
    script_has_arg: bool = True  # 该上标/下标之后是否已有完整参数
    script_just_completed: bool = False  # 是否刚完成一个脚本参数（本步新完成）

    def update(self, new_text: str):
        if not new_text:
            return
        self.generated_text += new_text
        self._update_script_state(new_text)
        self._update_brackets(new_text)
        self._update_envs(new_text)

    def _update_script_state(self, text: str):
        self.script_just_completed = False  # 每次 update 先重置

        for ch in text:
            if ch in ('^', '_'):
                self.last_script_char = ch
                self.script_has_arg = False
                self.script_just_completed = False
            elif not self.script_has_arg:
                if ch == '{':
                    pass  # 等 } 闭合
                elif ch.strip():
                    # 单字符参数完成
                    self.script_has_arg = True
                    self.script_just_completed = True
            if ch == '}' and not self.script_has_arg:
                # {x} 形式的参数闭合
                self.script_has_arg = True
                self.script_just_completed = True

    def _update_brackets(self, text: str):
        for ch in text:
            if ch in OPEN_BRACKETS:
                self.bracket_stack.append(ch)
            elif ch in CLOSE_BRACKETS:
                expected = BRACKET_PAIRS[ch]
                if self.bracket_stack and self.bracket_stack[-1] == expected:
                    self.bracket_stack.pop()

    def _update_envs(self, text: str):
        for m in re.finditer(r'\\begin\{(\w+)\}', text):
            env = m.group(1)
            self.env_stack.append(env)
            if env in MATRIX_ENVS:
                self.in_matrix_env = True
        for m in re.finditer(r'\\end\{(\w+)\}', text):
            env = m.group(1)
            if self.env_stack and self.env_stack[-1] == env:
                self.env_stack.pop()
            self.in_matrix_env = any(e in MATRIX_ENVS for e in self.env_stack)


# ================================================================
# LatexConstraintProcessor
# ================================================================

class LatexConstraintProcessor(LogitsProcessor):

    def __init__(
        self,
        tokenizer=None,
        model=None,
        penalty: float = -float('inf'),
        verbose: bool = False,
    ):
        # 支持从 model.name_or_path 自动加载 tokenizer
        if tokenizer is None:
            assert model is not None, (
                "请传入 tokenizer 或 model：\n"
                "  LatexConstraintProcessor(tokenizer=tokenizer)\n"
                "  LatexConstraintProcessor(model=model)"
            )
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model.name_or_path, trust_remote_code=True
            )

        self.tokenizer = tokenizer
        self.penalty = penalty
        self.verbose = verbose

        # EOS / 特殊 token
        self._eos_token_id = tokenizer.eos_token_id
        special_ids = set(tokenizer.all_special_ids)
        for attr in ('eos_token_id', 'bos_token_id', 'pad_token_id', 'unk_token_id'):
            val = getattr(tokenizer, attr, None)
            if val is not None:
                special_ids.add(val)
        self._special_token_ids = list(special_ids)

        # 预计算词表
        self._vocab_decoded, self._static_banned = self._precompute_vocab(tokenizer, special_ids)

        # 预计算动态检查索引
        self._precompute_token_indices()

        # batch 状态（延迟初始化）
        self.states: list[LatexState] = []

        print(
            f"[LatexConstraintProcessor] "
            f"特殊token: {len(special_ids)}个  "
            f"静态过滤: {len(self._static_banned)}个  "
            f"动态检查词表: {len(self._vocab_decoded)}个"
        )

    # ── 词表预计算 ──────────────────────────────────────────────

    def _is_legal_latex_char(self, text: str) -> bool:
        """判断文本是否只含合法 LaTeX 字符"""
        LEGAL_PATTERN = re.compile(
            r'^['
            r'a-zA-Z0-9'            # 英文字母和数字
            r'\s'                   # 空白字符（空格、换行、tab）
            r'\\{}()\[\]'           # LaTeX 核心符号
            r'\+\-\*/=<>!&|^~'      # 运算符
            r'_\^'                  # 上下标
            r'.,;:\'"`'             # 标点
            r'#%'                   # 其他常用  r'@#%'，按数据集改为 r'#%' 
            r']+$'
        )
        ILLEGAL_RANGES = [
            (0x4E00, 0x9FFF),       # 中文基本汉字
            (0x3400, 0x4DBF),       # 中文扩展A
            (0x20000, 0x2A6DF),     # 中文扩展B
            (0xFF00, 0xFFEF),       # 全角字符
            (0x3000, 0x303F),       # 中文标点
            (0x0080, 0x009F),       # 不可见控制符
            (0xD800, 0xDFFF),       # UTF-16 代理区
            (0xE000, 0xF8FF),       # 私有区
        ]
        for char in text:
            cp = ord(char)
            for start, end in ILLEGAL_RANGES:
                if start <= cp <= end:
                    return False
        return bool(LEGAL_PATTERN.match(text))

    def _precompute_vocab(
        self,
        tokenizer,
        special_ids: set,
    ) -> tuple[dict[int, str], torch.Tensor]:
        """
        返回：
          vocab_decoded: 合法 token 的 id → 文本（用于动态检查）
          static_banned: 非法字符 token 的 id tensor（静态屏蔽）
        """
        vocab = tokenizer.get_vocab()
        static_banned = []
        vocab_decoded = {}

        for token, idx in vocab.items():
            # 特殊 token 直接保留，内容置空（不参与任何规则检查）
            if idx in special_ids:
                vocab_decoded[idx] = ""
                continue
            try:
                text = tokenizer.convert_tokens_to_string([token])
            except Exception:
                vocab_decoded[idx] = ""
                continue

            if not self._is_legal_latex_char(text):
                static_banned.append(idx)
            else:
                vocab_decoded[idx] = text

        return vocab_decoded, torch.tensor(static_banned, dtype=torch.long)

    def _precompute_token_indices(self):
        """
        按字符内容预分组，供动态屏蔽 O(1) 查找：
          _tokens_contain[ch] → 以字符 ch 开头的所有 token id 列表
          _end_env_ids[env]   → 包含 \\end{env} 的 token id 列表
          _cmd_token_ids      → 包含未知命令的 token id 列表（静态）
        """
        # 关键字符分组
        KEY_CHARS = ('^', '_', '&', ']', ')', '}')
        self._tokens_startwith: dict[str, list[int]] = {ch: [] for ch in KEY_CHARS}

        # \end{env} 分组
        self._end_env_ids: dict[str, list[int]] = {}

        # 非法命令（静态，加入 static_banned）
        invalid_cmd_ids = []

        for token_id, text in self._vocab_decoded.items():
            if not text:
                continue

            # 按包含字符分组
            for ch in KEY_CHARS:
                if text.lstrip().startswith(ch):
                    self._tokens_startwith[ch].append(token_id)

            # \end{env} 分组
            m = re.search(r'\\end\{(\w+)\}', text)
            if m:
                env = m.group(1)
                self._end_env_ids.setdefault(env, []).append(token_id)

            # 非法命令检查
            commands = re.findall(r'\\([a-zA-Z]+)', text)
            if any(cmd not in VALID_COMMANDS for cmd in commands):
                invalid_cmd_ids.append(token_id)

        # 非法命令并入静态禁止
        if invalid_cmd_ids:
            extra = torch.tensor(invalid_cmd_ids, dtype=torch.long)
            self._static_banned = torch.cat([self._static_banned, extra]).unique()

    # ── 动态屏蔽 ────────────────────────────────────────────────

    def _get_context_banned(self, state: LatexState) -> list[int]:
        """根据当前状态返回需要动态屏蔽的 token id"""
        banned = []

        # 规则1：^ 或 _ 之后尚无完整参数，禁止再出现 ^ 和 _
        if not state.script_has_arg:
            banned.extend(self._tokens_startwith['^'])
            banned.extend(self._tokens_startwith['_'])

        # 规则2：刚完成一个脚本参数（如 a^1 或 a^{12} 末尾），
        # 则禁止紧接着生成同类脚本（防止 a^1^ 或 a_1_2）
        # 注意：不禁止另一类，a^1_2 是合法的
        elif state.script_just_completed and state.last_script_char:
            banned.extend(self._tokens_startwith[state.last_script_char])

        # 规则3：不在矩阵环境，禁止 &
        if not state.in_matrix_env:
            banned.extend(self._tokens_startwith['&'])

        # 规则4：括号栈顶不匹配，禁止对应非法闭合括号
        if state.bracket_stack:
            top = state.bracket_stack[-1]
            for illegal_close in BRACKET_MISMATCH.get(top, ()):
                banned.extend(self._tokens_startwith[illegal_close])
        else:
            # 无开括号则禁止所有闭合括号
            for illegal_close in CLOSE_BRACKETS:
                banned.extend(self._tokens_startwith[illegal_close])

        # 规则5：\end{env} 必须与栈顶匹配
        if state.env_stack:
            top_env = state.env_stack[-1]
            for env, ids in self._end_env_ids.items():
                if env != top_env:
                    banned.extend(ids)

        return banned

    # ── __call__ ────────────────────────────────────────────────

    def __call__(
        self,
        input_ids: torch.LongTensor,   # [batch, seq_len]
        scores: torch.FloatTensor,     # [batch, vocab_size]
    ) -> torch.FloatTensor:

        batch_size = input_ids.shape[0]
        device = scores.device

        # 延迟初始化 / batch size 变化时重置状态
        if len(self.states) != batch_size:
            self.states = [LatexState() for _ in range(batch_size)]

        # 设备对齐（只在首次或设备变化时迁移）
        if self._static_banned.device != device:
            self._static_banned = self._static_banned.to(device)

        for i in range(batch_size):
            # 已生成 EOS 的样本跳过
            if self._eos_token_id is not None:
                if (input_ids[i] == self._eos_token_id).any():
                    continue

            # 更新第 i 个样本的状态
            if input_ids.shape[1] > 0:
                last_id = input_ids[i, -1].item()
                last_text = self._vocab_decoded.get(last_id, "")
                self.states[i].update(last_text)

                if self.verbose:
                    print(
                        f"[batch={i}] last_token={last_text!r}  "
                        f"env_stack={self.states[i].env_stack}  "
                        f"bracket_stack={self.states[i].bracket_stack}  "
                        f"in_matrix={self.states[i].in_matrix_env}"
                    )

            # 静态屏蔽
            scores[i, self._static_banned] = self.penalty

            # 动态屏蔽
            context_banned = self._get_context_banned(self.states[i])
            if context_banned:
                scores[i, context_banned] = self.penalty

        # 强制恢复所有特殊 token（确保 EOS 不被屏蔽）
        for sp_id in self._special_token_ids:
            scores[:, sp_id] = scores[:, sp_id].clamp(min=0)

        return scores

    def reset(self):
        """每次新的 generate 调用前重置所有状态"""
        self.states = []


if __name__ == "__main__":

    from transformers import AutoTokenizer, AutoModel
    # model = AutoModel.from_pretrained('/root/dev/ckpts/ft4_postprocess', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained('/root/dev/ckpts/ft4_postprocess', trust_remote_code=True)
    processor = LatexConstraintProcessor(tokenizer=tokenizer)