import os
import subprocess
import requests
import sys
import threading
import pkg_resources
from typing import Optional


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

    # --- LLM Interaction ---
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
            result = requests.post(url, json=payload, headers=headers)
            result.raise_for_status()
            data = result.json()
            return data["choices"][0]["message"]["content"]

        elif self.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={os.environ.get('GEMINI_API_KEY')}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            headers = {"Content-Type": "application/json"}
            result = requests.post(url, json=payload, headers=headers)
            result.raise_for_status()
            data = result.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

        else:
            raise ValueError("Unknown provider. Use 'perplexity' or 'gemini'.")

    # --- Code Extraction ---
    def response_to_pycode(self, response: str) -> Optional[str]:
        for marker in ["```python", "```"]:
            start_index = response.find(marker)
            if start_index != -1:
                start_index += len(marker)
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

    # --- Dependency Handling ---
    @staticmethod
    def parse_imports(code: str):
        imports = set()
        for line in code.splitlines():
            line = line.strip()
            if line.startswith("import "):
                imports.add(line.split()[1].split(".")[0])
            elif line.startswith("from "):
                imports.add(line.split()[1].split(".")[0])
        return sorted(imports)

    def install_missing_packages(self, packages):
        installed = {pkg.key for pkg in pkg_resources.working_set}
        to_install = [p for p in packages if p.lower() not in installed]
        if not to_install:
            print("[INFO] All dependencies already installed.")
            return 0
        print(f"[INFO] Installing missing packages: {to_install}")
        process = subprocess.Popen(
            [sys.executable, "-m", "pip", "install"] + to_install,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate()
        print(stdout)
        if stderr:
            print(stderr)
        return process.returncode

    def dependency_manager(self, code):
        imports = self.parse_imports(code)
        if not imports:
            print("[INFO] No imports found — skipping dependency install.")
            return 0
        return self.install_missing_packages(imports)

    # --- Script Execution ---
    def run_script_realtime(self, filepath: str = "./outputs/pycode.py"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found.")

        process = subprocess.Popen(
            [sys.executable, filepath],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )

        def stream_output(pipe, is_err=False):
            for line in iter(pipe.readline, ''):
                if is_err:
                    print(f"[stderr] {line}", end='')
                else:
                    print(line, end='')
            pipe.close()

        threading.Thread(target=stream_output, args=(process.stdout, False)).start()
        threading.Thread(target=stream_output, args=(process.stderr, True)).start()
        process.wait()
        return process.returncode

    # --- Debug Loop ---
    def __call__(self, prompt: str):
        prompt_with_rule = prompt.strip() + "\n\nIMPORTANT: Return ONLY a valid Python code block."
        response = self.generate(prompt_with_rule)
        print("[INFO] Generated response (truncated):")
        print(response[:500] + ("..." if len(response) > 500 else ""))

        self.response_to_pyfile(response)
        with open("./outputs/pycode.py", "r", encoding="utf-8") as f:
            code = f.read()

        self.dependency_manager(code)
        retcode = self.run_script_realtime()

        attempt = 0
        while retcode != 0 and attempt < 10:
            attempt += 1
            print(f"[ERROR] Script exited with code {retcode}. Attempting auto-debug fix #{attempt}...")

            with open("./outputs/pycode.py", "r", encoding="utf-8") as f:
                code = f.read()

            debug_prompt = (
                f"You are a Debugger. Fix the Python code based on this error.\n"
                f"Error: Script exited with code {retcode}\n"
                f"Source Code:\n{code}\n"
                f"Original Prompt: {prompt}\n"
                f"IMPORTANT: Return ONLY a valid Python code block and it should be the FULL ENTIRE CODE, not just a fix."
            )

            # Target known recurring errors
            if "max_rows" in code and "DataFrame" in code:
                debug_prompt += "\n\nNote: In latest Gradio, gr.DataFrame does not have 'max_rows'. Remove it or replace with supported args like 'row_count'."

            debug_response = self.generate(debug_prompt)
            print("[INFO] Generated debug fix (truncated):")
            print(debug_response[:500] + ("..." if len(debug_response) > 500 else ""))

            pycode = self.response_to_pycode(debug_response)
            if not pycode:
                raise ValueError("Debug fix did not return valid Python code.")

            with open("./outputs/pycode.py", "w", encoding="utf-8") as f:
                f.write(pycode)

            self.dependency_manager(pycode)
            retcode = self.run_script_realtime()

        if retcode == 0:
            print("[INFO] Script executed successfully!")
        else:
            print(f"[ERROR] Script still failed after {attempt} debug attempts.")

        return retcode