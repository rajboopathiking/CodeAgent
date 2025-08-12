import os
import subprocess
import requests
import time
from typing import Optional, List

class CodeAgent:
    def __init__(self, apikey: str):
        self.apikey = apikey
        if not os.environ.get("PPLX_API_KEY"):
            os.environ["PPLX_API_KEY"] = self.apikey
        os.makedirs("./outputs/", exist_ok=True)

    def generate(self, prompt: str) -> str:
        """Send prompt to Perplexity API and return the raw response text."""
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

        result = requests.post(url, json=payload, headers=headers)
        try:
            data = result.json()
        except Exception:
            raise RuntimeError(f"API did not return valid JSON. Status: {result.status_code}, Text: {result.text}")

        return data["choices"][0]["message"]["content"]

    def response_to_pycode(self, response: str) -> Optional[str]:
        """Extract Python code from LLM markdown response."""
        start_marker = "```python"
        start_index = response.find(start_marker)
        if start_index == -1:
            return None
        start_index += len(start_marker)
        end_marker = "```"
        end_index = response.find(end_marker, start_index)
        if end_index == -1:
            return response[start_index:].strip()
        return response[start_index:end_index].strip()

    def save_pyfile(self, code: str):
        """Save code to pycode.py."""
        with open("./outputs/pycode.py", "w") as f:
            f.write(code)

    def run_code(self):
        """Run pycode.py and stream stdout/stderr in real-time with monitoring."""
        process = subprocess.Popen(
            ["python", "./outputs/pycode.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        stdout_lines, stderr_lines = [], []
        start_time = time.time()

        # Real-time output display
        while True:
            stdout_line = process.stdout.readline()
            if stdout_line:
                print(f"[STDOUT] {stdout_line.strip()}")
                stdout_lines.append(stdout_line)

            stderr_line = process.stderr.readline()
            if stderr_line:
                print(f"[STDERR] {stderr_line.strip()}")
                stderr_lines.append(stderr_line)

            if process.poll() is not None:
                break

        elapsed = time.time() - start_time
        process.wait()
        return process.returncode, stdout_lines, stderr_lines, elapsed

    def Workflow(self, prompt: str, max_retries: int = 10):
        """Main loop: generate → run → fix until success with transparency."""
        print("🚀 Generating initial code...")
        initial_code = self.response_to_pycode(self.generate(prompt))
        if not initial_code:
            raise ValueError("❌ LLM did not return valid Python code.")
        self.save_pyfile(initial_code)

        for attempt in range(max_retries):
            print("\n" + "=" * 60)
            print(f"📜 Attempt {attempt + 1} — Current Code to Execute:")
            print("=" * 60)

            with open("./outputs/pycode.py") as f:
                source_code = f.read()
            print(source_code)
            print("=" * 60)

            returncode, stdout_lines, stderr_lines, elapsed = self.run_code()

            print(f"⏱️ Elapsed Time: {elapsed:.2f} seconds")
            print(f"🔚 Process finished with return code: {returncode}")

            if returncode == 0:
                print("✅ Success! Code executed without errors.")
                return 0

            print("🔄 Debugging with LLM...")
            debug_prompt = (
                f"You are a Python debugger.\n"
                f"Original task: {prompt}\n"
                f"Here is the source code that failed:\n{source_code}\n"
                f"STDOUT:\n{''.join(stdout_lines)}\n"
                f"STDERR:\n{''.join(stderr_lines)}\n"
                f"Please return the FULL corrected Python code only."
            )

            new_code = self.response_to_pycode(self.generate(debug_prompt))
            if not new_code:
                print("❌ LLM did not return valid Python code. Stopping.")
                break
            if new_code.strip() == source_code.strip():
                print("⚠️ Code unchanged after fix attempt, stopping.")
                break

            self.save_pyfile(new_code)

        print("❌ Max retries reached without success.")
        return 1