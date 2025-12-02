
import os, json, textwrap, re, time
import requests

API_KEY  = os.getenv("OPENAI_API_KEY", "cse476")
API_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")  
MODEL    = os.getenv("MODEL_NAME", "bens_model")              

def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.0,
                                timeout: int = 60) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    """
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 128,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs   = dict(resp.headers)
        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
            # try best-effort to surface error text
            err_text = None
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}

ACTION_RE = re.compile(r"^\s*(CALCULATE|FINAL)\s*:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)

# --- PROVIDED: prompts ---
SYSTEM_AGENT = """
You are a reasoning agent that has to solve various different tasks.
Here are the list of tasks:
1) common sense question and answering
2) future prediction (even if it's impossible, just follow the question's instructions), 
3) coding tasks (output only valid code when asked)
4) math problems
5) planning tasks (output plans in the required format)

You have access to one tool:
1) CALCULATE: <arithmetic expression>
   - use only numbers, + - * / **, parentheses, and round(x, ndigits)
   - example: CALCULATE: round((3*2.49)*1.07, 2)

Or you can give the final answer in this format:
2) FINAL: <answer>

When solving a task, make sure to follow these guidelines:
1) Read the user question carefully and follow formatting instructions in it
2) When you choose FINAL, <answer> should exactly be the final expected output (for example: include \\boxed{...}, or only code, or only the plan lines) and nothing else.
3) Return ONE line only: either
  CALCULATE: <expression>
  or
  FINAL: <answer>
  Make sure not to include further explanations, reasoning steps, or extra text.
"""

def make_first_prompt(question: str) -> str:
    return f"""You will receive a task that may be under these domains: common sense, math, coding, planning, or future prediction.

Task:
{question}

If you need arithmetic to complete the task (for example, computing an intermediate numeric value), reply as:
CALCULATE: <expression>

If you DO NOT need to do arithmetic, or once you have all needed numeric results, then reply as:
FINAL: <answer>"""

def make_second_prompt(result: str) -> str:
    return f"""The calculation result from the CALCULATE tool is: {result}

Provide the FINAL answer to the original task, replying in this format exactly:
FINAL: <answer>"""

def parse_action(text: str):
    """
    Returns ("CALCULATE", expr) or ("FINAL", answer); raises ValueError on bad format.
    """
    m = ACTION_RE.match(text.strip())
    if not m:
        raise ValueError(f"Unrecognized action format: {text!r}")
    action = m.group(1).upper()
    payload = m.group(2).strip()
    return action, payload


def calculator(expr: str):
    allowed_names = {"round": round}
    return eval(expr, {"__builtins__": {}}, allowed_names)


#===========================================================================
# Techniques to implement:
# 1. tool reasoning with calculate (done)
# 2. classify domain type: 
#   math, common sense, coding, planning, future prediction
# 3. CoT for these domains: math and planning
# 4. self-verify
#===========================================================================

def run_agent(question: str, max_tool_uses: int = 2, verbose: bool = True):
    r1 = call_model_chat_completions(prompt=make_first_prompt(question), system=SYSTEM_AGENT, temperature=0.0,)
    if not r1["ok"]:
        raise RuntimeError(f"API error: {r1['error']}")

    if verbose: print("LLM →", r1["text"])
    action, payload = parse_action(r1["text"])

    tool_uses = 0

    while action == "CALCULATE":
        if tool_uses >= max_tool_uses:
            raise RuntimeError("Exceeded tool-use limit.")
        tool_uses += 1

        calc_value = calculator(payload)
        if verbose: print("CALC =", calc_value)

        #ask model again with calculator result
        rN = call_model_chat_completions(prompt=make_second_prompt(str(calc_value)), system=SYSTEM_AGENT, temperature=0.0,)
        if not rN["ok"]:
            raise RuntimeError(f"API error: {rN['error']}")
        if verbose: print("LLM →", rN["text"])

        action, payload = parse_action(rN["text"])

    # action must be FINAL here
    return payload