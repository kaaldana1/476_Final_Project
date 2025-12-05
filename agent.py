from collections import Counter
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

#===========================================================================
# Techniques to implement:
# 1. tool reasoning with calculate (done)
# 2. classify domain type: 
#   math, common sense, coding, planning, future prediction
# 3. CoT for these domains: math and planning
# 4. self-verify
#===========================================================================

#===========================================================================
# Domain classifier: math, common sense, coding, planning or future preducition
#===========================================================================

def classify_domain(question:str) -> str:
    question = question.lower()

    coding_task_phrases = ['import', 'def', 'write a function', 'implement', 'class', 'script', 'program', 'python', 'write code']
    
    planning_task_phrases = ['[PLAN]', 'My plan is as follows']
    
    # dev prompt is in the style "you are an agenet that can predict future events"
    future_prediction_phrase = "You are an agent that can predict future events."

    math_task_phrases = ['calculate', 'solve', 'compute', 'what is', 'determine', 'evaluate', 
                         'find the value', 'how much', 'how many', 'estimate', 
                         'total', 'sum', 'difference', 'product', 'quotient', 'percentage', 'increase', 'decrease',
                         'percentage', 'area', 'volume', 'length', 'distance', 'angle', 'equation', 'formula', 
                         'coordinates', 'expression', 'integral', 'derivative', 'function', 'graph', 'plot',
                         'log_', 'ln(', 'e^', 'pi', 'sqrt(']

    if any(phrase in question for phrase in coding_task_phrases):
        return "coding"
    elif any(phrase in question for phrase in planning_task_phrases):
        return "planning"
    elif future_prediction_phrase in question:
        return "future prediction"
    elif any(phrase in question for phrase in math_task_phrases):
        return "math"
    else:
        return "common sense"
    

def make_first_prompt(question: str, domain: str) -> str:
    if domain == "math":
        return f"""You are a reasoning agent that has to solve a math problem.
Below is a set of guidelines to follow; however, if the task instructions contradict any of these guidelines, ALWAYS prioritize the task instructions.
Math-specific Guidelines:
- If you need arithmetic to complete the task (for example, computing an intermediate numeric value), reply as:
CALCULATE: <expression>
- If you DO NOT need to do arithmetic, or once you have all needed numeric results, then reply in this format:
FINAL: <answer>

Task:
{question}
"""
    elif domain == "planning":
        return f"""You are a reasoning agent that has to solve a STRIPS planning task. 

Below is a set of guidelines to follow; however, if the task instructions contradict any of these guidelines, ALWAYS prioritize the task instructions.
Planning-specific Guidelines:
Return exactly this format, with no extra text:
FINAL: (action arg1 arg2 ...)\n(action arg1 arg2 ...)\n...
- You need to start with `FINAL: ` followed by the plan
- One action per line. Each action and its arguments are parenthesized.
- All letters hsould be lowercase
- Arguments should be in order.
- No numbering, no explanations or reasoning steps.

Task:
{question}
"""
    #=============================================================================
    elif domain == "coding":
        return f"""You are a reasoning agent that has to write a self-contained code solution for a coding task. 

Below is a set of guidelines to follow; however, if the task instructions contradict any of these guidelines, ALWAYS prioritize the task instructions.
Coding-specific Guidelines:
Return exactly this format, with no extra text:
FINAL: <code only, no extra text> 
- Include all required imports and helper functions inside the answer
- The code should be valid and should be able to run error-free
- Do not comment the code unless otherwise specified
- You MUST follow any signature/skeleton code or code instructions provided in the question verbatim
- Do NOT invent formatting that is not specified in the task
Task:
{question}
"""
    #=============================================================================
    elif domain == "future prediction":
        return f"""You are a reasoning agent that has to predict future events that is described in the task. Do not refuse.

Below is a set of guidelines to follow; however, if the task instructions contradict any of these guidelines, ALWAYS prioritize the task instructions.
Future prediction-specific Guidelines:
Return exactly this format, with no extra text:
FINAL: \\boxed{{YOUR_PREDICTION_HERE}}
- Output only one line starting with the "FINAL: " prefix
- Enclose your prediction in \\boxed{{...}} 
- Keep predictions concise and to the point
- No extra text, reasoning steps, or explanations
- Do NOT invent formatting that is not specified in the task

Task:
{question}
"""
    #=============================================================================
    elif domain == "common sense":
        return f"""You are a reasoning agent that has to solve a common sense question answering task. The single answer you provide must be the best one.

Below is a set of guidelines to follow; however, if the task instructions contradict any of these guidelines, ALWAYS prioritize the task instructions.
Common sense-specific Guidelines:
Return exactly this format, with no extra text:
FINAL: <answer>
- Make sure to provide only a ONE LINE answer starting with the "FINAL: " prefix
- If multiple words are necessary, keep them on the same line
- No extra text, reasoning steps, or explanations
- Do NOT invent formatting that is not specified in the task

Task:
{question}
"""
    else:
        return f""" """

# Some none-math domains may still require calculation. The second prompt 
# should remind the model of the domain and the format expected with that domain
def make_second_prompt(result: str, domain: str) -> str:
    
    if domain == "coding":
        reminder = """REMEMBER that you are solving a coding task. """
    elif domain == "planning":
        reminder = """REMEMBER that you are solving a STRIPS planning task. """
    elif domain == "future prediction":
        reminder = """REMEMBER that you are solving a future prediction task. """
    elif domain == "common sense":
        reminder = """REMEMBER that you are solving a common sense question and answering task."""
    else: 
        reminder = """REMEMBER that you are solving a math problem. """

    return f"""{reminder} 
        Your final answer should be valid code in the required format that follows the {domain} guidelines provided earlier exactly; 
        HOWEVER, if the task instructions contradict any of these guidelines, ALWAYS prioritize the task instructions.
The calculation result from the CALCULATE tool is: {result}
Now provide the final answer.
Reply exactly as: FINAL: <answer>"""



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
# Chain of thought: 
# Runs only for math and planning domains
# We make three separate passes
# each with prompts of temp 0.2 to produce more diverse reasoning
#===========================================================================

def single_pass_cot(question: str, previous_answer: str, system: str, domain: str, temperature: float = 0.2, verbose: bool = True) -> str:
    cot_prompt = f"""Your job: Think step by step about whether the proposed answer is consistent with the question and correct.
If you find any mistakes, your next step is to provide a corrected final answer in the required format that is specified in the task and {domain} guidelines. If the 
task requirements contradict the {domain} guidelines (provided earlier), ALWAYS prioritize the task requirements.

Question:
{question}
Proposed Final Answer:
{previous_answer}

Now provide the new final answer, do not include any reasoning steps or explanations in the final answer.
Reply exactly as: FINAL: <answer>"""

    
    cot = call_model_chat_completions(prompt=cot_prompt, system=system, temperature=temperature, max_tokens=256)
    if not cot["ok"]:
        raise RuntimeError(f"API error: {cot['error']}")

    if verbose: print("Verifier →", cot["text"])
    action, payload = parse_action(cot["text"])
    if action != "FINAL":
        return previous_answer
    return payload.strip()
    
def chain_of_thought(question: str, previous_answer: str, domain: str) -> str:
    if domain == "math":
        system = "You are a correctness verifier for math problems"
    else: #planning
        system = "You are a correctness verifier for STRIPS planning tasks"
    
    cot_answers = []
    for i in range(3):
        cot_answer = single_pass_cot(question, previous_answer, system, domain, temperature=0.2, verbose=False)
        cot_answers.append(cot_answer)

    counts = Counter(cot_answers)
    top_answer, top_count = counts.most_common(1)[0]
    return top_answer.strip()


#===========================================================================
# Self-verification: Tasks for any domain should require the model to self-verify
# that its final answer meets the task requirements and formatting instructions
# It does so by reading the task and the model's previous final answer, and
# asking the model to verify that the answer meets all requirements
#===========================================================================

def self_verification(question: str, previous_answer: str, domain: str, verbose: bool = True) -> str:
    system = "You are a strict answer-format validator."
    prompt = """You will receive a question and a proposed final answer.
    
Your job is to:
1) Read the task and take note any explicit output formatting instructions
   (for example: output must be in \\boxed{{...}}, or YES/NO answers only,
   or the output fraction must be simplified, or only Python code that builds upon (not remove) the base code, etc...)
2) Re-read the {domain}-specific guidelines provided earlier and take note of the the formatting instructions there as well
3) Read the proposed answer. If the proposed answer already follows the instructions and guidelines, 
    and is logically consistent, keep it as-is.
4) If there are any contradictions between the task instructions and the {domain}-specific guidelines, ALWAYS prioritize the task instructions.
Meaning if the answer already meets the task instructions but not the guidelines, keep it as-is.
4) Otherwise, modify the answer minimally so that it satisfies the instructions or guidelines and is more likely to be correct.
    Check whether the proposed Final Answer follows all the formatting instructions 
    and requirements that were specified in both the task below and the {domain}-specific guidelines provided earlier.

Task:
{question}
Previous Final Answer:
{previous_answer}

Keep your reasoning to the point and concise.

IMPORTANT:
- Do NOT explain your reasoning.
- Your entire reply must be in the form:
  FINAL: <correctly formatted answer>
"""
    format_verification = call_model_chat_completions(prompt=prompt, system=system, temperature=0.0, max_tokens=256)
    if not format_verification["ok"]:
        raise RuntimeError(f"API error: {format_verification['error']}")

    if verbose: print("Verifier →", format_verification["text"])
    action, payload = parse_action(format_verification["text"])
    if action != "FINAL":
        return previous_answer
    return payload.strip()



# ============================ MAIN AGENT LOOP ============================

def run_agent(question: str, max_tool_uses: int = 2, verbose: bool = True):
    # classify the domain
    domain = classify_domain(question)

    r1 = call_model_chat_completions(prompt=make_first_prompt(question, domain), system=SYSTEM_AGENT, temperature=0.0,)
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

if __name__ == "__main__":
    # Example usage
    question = "If you have 3 apples and you buy 2 more, how many apples do you have in total?"
    answer = run_agent(question, verbose=True)
    print("Final Answer:", answer)