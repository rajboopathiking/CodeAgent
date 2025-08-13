import os
import subprocess
import requests
import re
import autopep8
import sys
from typing import Optional
import threading
import pkg_resources


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
        """Send prompt to API and return the response text."""
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
        """Extract Python code from a model's response."""
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
        """Save extracted Python code to a file."""
        pycode = self.response_to_pycode(response)
        if pycode is None:
            raise ValueError("No python code block found in response.")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(pycode)
        print(f"[INFO] Saved extracted python code to {filename}")

    @staticmethod
    def parse_imports(code: str):
        """Extract top-level imports from Python code."""
        imports = set()
        for line in code.splitlines():
            line = line.strip()
            if line.startswith("import "):
                parts = line.split()
                if len(parts) >= 2:
                    imports.add(parts[1].split(".")[0])
            elif line.startswith("from "):
                parts = line.split()
                if len(parts) >= 2:
                    imports.add(parts[1].split(".")[0])
        return sorted(imports)

    def install_missing_packages(self, packages):
        """Install missing packages via pip."""
        installed = {pkg.key for pkg in pkg_resources.working_set}
        to_install = [p for p in packages if p.lower() not in installed]
        if not to_install:
            print("[INFO] All dependencies already installed.")
            return 0

        print(f"[INFO] Installing missing packages: {to_install}")
        process = subprocess.Popen(
            [sys.executable, "-m", "pip", "install"] + to_install,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate()
        print(stdout)
        if stderr:
            print(stderr)
        return process.returncode

    def dependency_manager(self, code):
        """Extract imports and install missing dependencies."""
        imports = self.parse_imports(code)
        if not imports:
            print("[INFO] No imports found — skipping dependency install.")
            return 0

        # Step 1: Install directly from imports
        ret = self.install_missing_packages(imports)
        if ret != 0:
            print("[WARN] Some dependencies from imports failed. Asking LLM for requirements...")
            # Step 2: Ask LLM for requirements if needed
            content = self.generate(
                f"Generate the pip install requirements list for the following code:\n{code}\n"
                f"Only output package names (and versions if known) in a Python code block."
            )
            reqs = self.response_to_pycode(content)
            if reqs:
                with open("./outputs/requirements.txt", "w") as f:
                    f.write(reqs)
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", "./outputs/requirements.txt"],
                    check=False
                )
        return 0

    def run_script_realtime(self, filepath: str = "./outputs/pycode.py"):
        """Run Python script and stream output in real-time."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found.")

        process = subprocess.Popen(
            [sys.executable, filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
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
        """Generate, run, and debug Python code."""
        prompt_with_rule = prompt.strip() + "\n\nIMPORTANT: Return ONLY a valid Python code block."
        response = self.generate(prompt_with_rule)
        print("[INFO] Generated response (truncated):")
        print(response[:500] + ("..." if len(response) > 500 else ""))

        self.response_to_pyfile(response, filename="./outputs/pycode.py")

        with open("./outputs/pycode.py", "r", encoding="utf-8") as f:
            code = f.read()

        self.dependency_manager(code)
        retcode = self.run_script_realtime("./outputs/pycode.py")

        if retcode != 0:
            print(f"[ERROR] Script exited with code {retcode}. Attempting auto-debug fix...")

            with open("./outputs/pycode.py", "r", encoding="utf-8") as f:
                code = f.read()

            debug_prompt = (
                f"You are a Debugger. Fix the Python code based on this error.\n"
                f"Error: Script exited with code {retcode}\n"
                f"Source Code:\n{code}\n"
                f"Original Prompt: {prompt}\n"
                f"IMPORTANT: Return ONLY a valid Python code block.and It should Be full Entire Code not just Fix"
            )
            debug_response = self.generate(debug_prompt)
            print("[INFO] Generated debug fix (truncated):")
            print(debug_response[:500] + ("..." if len(debug_response) > 500 else ""))

            pycode = self.response_to_pycode(debug_response)
            if not pycode:
                raise ValueError("Debug fix did not return valid Python code.")

            with open("./outputs/pycode.py", "w", encoding="utf-8") as f:
                f.write(pycode)

            self.dependency_manager(pycode)
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