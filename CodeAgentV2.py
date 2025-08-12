import os
import subprocess
import requests
import tempfile
import re
import autopep8
from typing import Optional
import threading


class CodeAgent:
    def __init__(self, pplx_apikey: Optional[str] = None, gemini_apikey: Optional[str] = None, provider: str = "perplexity"):
        self.provider = provider.lower()
        self.pplx_apikey = pplx_apikey
        self.gemini_apikey = gemini_apikey

        if self.pplx_apikey and not os.environ.get("PPLX_API_KEY"):
            os.environ["PPLX_API_KEY"] = self.pplx_apikey
        if self.gemini_apikey and not os.environ.get("GEMINI_API_KEY"):
            os.environ["GEMINI_API_KEY"] = self.gemini_apikey

        os.makedirs("./outputs/", exist_ok=True)

    def generate(self, prompt: str) -> str:
        if self.provider == "perplexity":
            url = "https://api.perplexity.ai/chat/completions"
            payload = {
                "model": "sonar-pro",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50000
            }
            headers = {
                "Authorization": f"Bearer {os.environ.get('PPLX_API_KEY')}",
                "Content-Type": "application/json"
            }
            try:
                result = requests.post(url, json=payload, headers=headers)
                result.raise_for_status()
            except requests.RequestException as e:
                raise RuntimeError(f"Perplexity API request failed: {e}")
            data = result.json()
            return data["choices"][0]["message"]["content"]

        elif self.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={os.environ.get('GEMINI_API_KEY')}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            headers = {"Content-Type": "application/json"}
            try:
                result = requests.post(url, json=payload, headers=headers)
                result.raise_for_status()
            except requests.RequestException as e:
                raise RuntimeError(f"Gemini API request failed: {e}")
            data = result.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

        else:
            raise ValueError("Unknown provider. Use 'perplexity' or 'gemini'.")

    def response_to_pycode(self, response: str) -> Optional[str]:
        start_marker = "```python"
        start_index = response.find(start_marker)
        if start_index != -1:
            start_index += len(start_marker)
            end_marker = "```"
            end_index = response.find(end_marker, start_index)
            if end_index != -1:
                return response[start_index:end_index].strip()

        start_marker = "```"
        start_index = response.find(start_marker)
        if start_index != -1:
            start_index += len(start_marker)
            end_index = response.find("```", start_index)
            if end_index != -1:
                return response[start_index:end_index].strip()

        if response.strip():
            return response.strip()
        return None

    def response_to_pyfile(self, response: str, filename: str = "./outputs/pycode.py"):
        pycode = self.response_to_pycode(response)
        if pycode is None:
            raise ValueError("No python code block found in response.")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(pycode)
        print(f"[INFO] Saved extracted python code to {filename}")

    @staticmethod
    def split_into_cells(code: str):
        """
        Split Python code into logical cells while preserving decorators, docstrings,
        and class/method blocks with indentation.
        """
        lines = code.splitlines()
        cells = []
        buffer = []
        pending_decorators = []
        in_block = False
        base_indent = None
        in_triple_quote = False
        triple_quote_delim = None

        for idx, line in enumerate(lines):
            stripped = line.strip()

            # Handle decorators
            if stripped.startswith("@"):
                pending_decorators.append(line)
                continue

            # Track triple quotes
            if not in_triple_quote and (stripped.startswith('"""') or stripped.startswith("'''")):
                triple_quote_delim = stripped[:3]
                in_triple_quote = True
            elif in_triple_quote and triple_quote_delim in stripped:
                in_triple_quote = False

            # Detect new block
            is_block_start = bool(re.match(r'^(class |def |async def )', stripped))
            if stripped.startswith("@dataclass") and idx + 1 < len(lines):
                nxt = lines[idx + 1].lstrip()
                if nxt.startswith("class "):
                    pending_decorators.append(line)
                    continue

            if is_block_start and not in_block:
                if buffer:
                    cells.append("\n".join(buffer).strip())
                    buffer = []
                if pending_decorators:
                    buffer.extend(pending_decorators)
                    pending_decorators = []
                buffer.append(line)
                in_block = True
                base_indent = len(line) - len(line.lstrip())
                continue

            if in_block:
                current_indent = len(line) - len(line.lstrip())
                if in_triple_quote or stripped == "" or current_indent > base_indent:
                    buffer.append(line)
                    continue
                else:
                    cells.append("\n".join(buffer).strip())
                    buffer = []
                    in_block = False
                    base_indent = None

            if not in_block:
                buffer.append(line)

        if buffer:
            cells.append("\n".join(buffer).strip())

        return [c for c in cells if c.strip()]

    def create_instrumented_script(self, code: str):
        cells = self.split_into_cells(code)
        instrumented = []
        for i, cell in enumerate(cells, 1):
            cell = autopep8.fix_code(cell, options={'aggressive': 1})
            lines = cell.splitlines()
            non_empty_lines = [line for line in lines if line.strip()]
            if not non_empty_lines:
                continue
            min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
            dedented_lines = [line[min_indent:] if line.strip() else line for line in lines]
            dedented_cell = "\n".join(dedented_lines)
            instrumented.append(f'print("<<CELL {i} START>>")')
            instrumented.append(dedented_cell)
            instrumented.append(f'print("<<CELL {i} END>>")')
        return "\n\n".join(instrumented)

    def run_script_realtime(self, filepath: str = "./outputs/pycode.py"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found.")

        process = subprocess.Popen(
            ["python", filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )

        def stream_output(pipe, is_err=False):
            for line in iter(pipe.readline, ''):
                if is_err:
                    print(f"[stderr] {line}", end='')
                else:
                    print(line, end='')
            pipe.close()

        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, False))
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, True))
        stdout_thread.start()
        stderr_thread.start()

        process.wait()
        stdout_thread.join()
        stderr_thread.join()

        return process.returncode

    def __call__(self, prompt: str):
        response = self.generate(prompt)
        print("[INFO] Generated response (truncated):")
        print(response[:500] + ("..." if len(response) > 500 else ""))
        self.response_to_pyfile(response, filename="./outputs/pycode.py")

        retcode = self.run_script_realtime("./outputs/pycode.py")
        if retcode != 0:
            print(f"[ERROR] Script exited with code {retcode}. Attempting auto-debug fix...")

            with open("./outputs/pycode.py", "r", encoding="utf-8") as f:
                code = f.read()

            debug_prompt = (
                f"You are a Debugger. Fix the Python code based on this error.\n"
                f"Error: Script exited with code {retcode}\n"
                f"Source Code:\n{code}\n"
                f"Original Prompt: {prompt}"
            )
            debug_response = self.generate(debug_prompt)
            print("[INFO] Generated debug fix (truncated):")
            print(debug_response[:500] + ("..." if len(debug_response) > 500 else ""))
            self.response_to_pyfile(debug_response, filename="./outputs/pycode.py")

            retcode_fix = self.run_script_realtime("./outputs/pycode.py")
            if retcode_fix != 0:
                print(f"[ERROR] Debug fix failed to run script with exit code {retcode_fix}.")
                return retcode_fix
            else:
                print("[INFO] Debug fix script executed successfully!")
                return 0
        else:
            print("[INFO] Script executed successfully!")
            return 0