import re
import ast
import textwrap


def sanitize_simulation_code(code: str) -> str:
    """
    Cleans LLM output to ensure it's valid executable Python code:
    - Removes triple backticks
    - Removes markdown wrappers and explanations
    - Ensures REQUIREMENTS and simulate(...) function are present
    """
    # Remove any preamble like 'Here is the code:'
    code = re.sub(r"(?i)^.*here\s+is\s+the\s+code.*\n?", "", code.strip())
    # Remove any markdown or triple backticks
    code = code.replace("```python", "").replace("```", "")
    # Normalize smart quotes
    code = code.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    code = textwrap.dedent(code)  # remove common indent
    code = re.sub(r"^\s*\n", "", code, count=1)  # strip leading blank line
    return code.rstrip() + "\n"


_UNMATCHED_QUOTE   = re.compile(r'(^|[^\\])(")(?=[^"\n]*$)')     # unescaped " with no close on that line
_UNMATCHED_SQUOTE  = re.compile(r"(^|[^\\])(')(?=[^'\n]*$)")     # same for '

# def remove_unmatches_quotes(src: str) -> str:
#     """
#     • Strip triple-quoted blocks      (LLM doc-strings / explanations)
#     • Dedent & strip leading blanks
#     • Remove any unmatched single / double quotes that remain
#     • Fail fast if the cleaned code still won’t compile
#     """
#     # 1. remove all triple-quoted strings
#     parts, keep = re.split(r"'''|\"\"\"", src), True
#     no_triple   = "".join(p for p in parts if keep := not keep)
#
#     # 2. clean indent / blank top lines
#     cleaned = textwrap.dedent(no_triple).lstrip()
#
#     # 3. kill dangling quotes – line-by-line
#     fixed_lines = []
#     for line in cleaned.splitlines():
#         line = _UNMATCHED_QUOTE.sub(r"\1#",  line)   # replace with #  → turns it into a comment char
#         line = _UNMATCHED_SQUOTE.sub(r"\1#", line)
#         fixed_lines.append(line)
#     fixed = "\n".join(fixed_lines)
#
#     return fixed


def validate_simulation_code(code: str):
    try:
        print(code)
        tree = ast.parse(code)
        func_names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
        if "simulate" not in func_names:
            raise ValueError("Missing required simulate() function")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in simulation code: {e}")

_unterminated_pat = re.compile(r'(["\'])([^"\']*)$')   # open quote to EOL

def autofix_unterminated(code: str) -> str:
    """
    If the first 10 lines contain an unterminated string literal, append
    the missing quote at end-of-line.  Returns patched code.
    """
    lines = code.splitlines()
    changed = False
    for i in range(min(10, len(lines))):
        line = lines[i]
        m = _unterminated_pat.search(line)
        if m and not line.strip().startswith("#"):
            lines[i] = line + m.group(1)   # add the missing quote
            changed = True
    return "\n".join(lines) if changed else code

import re
from typing import Tuple

_TRIPLE_QUOTE = re.compile(r'("""|\'\'\')')       # both styles
_MARKDOWN     = re.compile(r'`{3,}')              # ``` or more
_FREE_TEXT    = re.compile(r'^[A-Za-z].*$')       # lone prose line

def strip_trailing_extras(code: str) -> Tuple[str, bool]:
    """
    • removes *everything* after the last valid Python statement
        (heuristic: the last line that ends with ':' or ')', ']', '}', 'pass', 'return', etc.)
    • comments-out orphan literals ( 0.1 , "foo" )
    • deletes triple-quoted blocks & stray markdown fences

    Returns (fixed_code, changed)
    """
    changed = False
    lines   = code.splitlines()

    # 1 ▸ zap triple-quoted regions and markdown fences
    cleaned = []
    skip_block = False
    for ln in lines:
        if _TRIPLE_QUOTE.search(ln):
            skip_block = not skip_block
            changed = True
            continue
        if skip_block or _MARKDOWN.search(ln):
            changed = True
            continue
        cleaned.append(ln)

    lines = cleaned

    # 2 ▸ comment-out orphan literals or plain English lines
    safe = []
    for ln in lines:
        s = ln.strip()
        if (s and                               # non-empty
            not s.startswith("#") and           # not a comment already
            not re.match(r'\w', s.split()[0])   # doesn’t start with def / import / REQUIREMENTS / etc.
           ):
            safe.append("# " + ln)              # comment it
            changed = True
        else:
            safe.append(ln)

    # 3 ▸ strip any trailing prose after the final } ] ) return pass
    for i in reversed(range(len(safe))):
        if re.search(r'(\)|\]|\}|pass|return)\s*$', safe[i].strip()):
            last_good = i
            break
    else:
        last_good = len(safe)-1

    if last_good < len(safe)-1:
        safe = safe[:last_good+1]
        changed = True

    return "\n".join(safe), changed
