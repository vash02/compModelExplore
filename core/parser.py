# core/parser.py
import json, re
from llm.local_llm import LocalLLM
from llm.prompt_templates import parser_prompt   # <- unchanged name

_missing_comma = re.compile(r'(":[^,{}\[\]]+)\s+"')  # value "  "next_key

def tidy_json(txt: str) -> str:
    """Common fixes: smart quotes, trailing / missing commas."""
    txt = txt.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    txt = re.sub(r",\s*}", "}", txt)
    txt = re.sub(r",\s*]", "]", txt)
    txt = _missing_comma.sub(r'\1, "', txt)
    return txt

def parse_nl_input(nl_text: str, retries: int = 4, temperature: float = 0.0) -> dict:
    """
    Calls the local LLM with:
        system_prompt = parser_prompt
        prompt        = nl_text
    Retries `retries` times if the JSON cannot be parsed.
    """
    llm = LocalLLM(model="phi3:3.8b") # default model: deepseek-coder

    for attempt in range(retries + 1):
        raw = llm.generate(
            prompt=nl_text,
            system_prompt=parser_prompt,
            temperature=temperature,
        ).strip()

        print(f"\n⛳ RAW LLM OUTPUT (attempt {attempt}):\n{raw[:300]}\n")

        # 1) strict parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 2) tidy + parse
            try:
                return json.loads(tidy_json(raw))
            except json.JSONDecodeError:
                if attempt < retries:
                    print("⚠️  JSON decode failed, retrying …")
                    continue
                raise ValueError(
                    f"LLM failed to return valid JSON after {retries+1} attempts.\nRaw output:\n{raw}"
                )
