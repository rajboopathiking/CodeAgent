# Requirements: requests pillow pypdf2 importlib-metadata
import os
import sys
import re
import ast
import threading
import subprocess
import requests
import pkg_resources
import importlib.util
import base64
from typing import Optional, Union, List, Callable
from PIL import Image
import PyPDF2
import io

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class CodeAgent:
    """
    A framework-style Code Agent that can:
      - Generate Python code from text prompts (remote API or local function).
      - Install missing dependencies automatically.
      - Debug failed code up to N retries.
      - Run scripts with real-time stdout/stderr streaming.

    Provider options: "openai", "anthropic", "gemini", "perplexity", or "local".
    """

    def __init__(self,
                 provider: str = "local",
                 model: Optional[str] = None,
                 local_fn: Optional[Callable[[str], str]] = None,
                 pplx_apikey: Optional[str] = None,
                 gemini_apikey: Optional[str] = None,
                 anthropic_apikey: Optional[str] = None,
                 openai_apikey: Optional[str] = None,
                 attempt_limit = 5):
        self.provider = provider.lower()
        self.model = model
        self.local_fn = local_fn
        self.attempt_limit = attempt_limit

        # Defaults for hosted providers
        self.default_models = {
            "perplexity": "sonar-pro",
            "gemini": "gemini-2.5-flash",
            "anthropic": "claude-3-5-sonnet-20241022",
            "openai": "gpt-4o"
        }
        if not self.model and self.provider in self.default_models:
            self.model = self.default_models[self.provider]

        # API keys (env fallback)
        if pplx_apikey and not os.environ.get("PPLX_API_KEY"):
            os.environ["PPLX_API_KEY"] = pplx_apikey
        if gemini_apikey and not os.environ.get("GEMINI_API_KEY"):
            os.environ["GEMINI_API_KEY"] = gemini_apikey
        if anthropic_apikey and not os.environ.get("ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = anthropic_apikey
        if openai_apikey and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = openai_apikey

        os.makedirs("./outputs/", exist_ok=True)

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------
    def generate(self, prompt: str) -> str:
        """
        Generate code/text from the provider or local function.
        """
        if self.provider == "local":
            if not self.local_fn:
                raise ValueError("Local provider requires a local_fn callback.")
            return self.local_fn(prompt)

        elif self.provider == "perplexity":
            url = "https://api.perplexity.ai/chat/completions"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 5000
            }
            headers = {
                "Authorization": f"Bearer {os.environ.get('PPLX_API_KEY')}",
                "Content-Type": "application/json"
            }
            result = requests.post(url, json=payload, headers=headers)
            result.raise_for_status()
            return result.json()["choices"][0]["message"]["content"]

        elif self.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={os.environ.get('GEMINI_API_KEY')}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            result = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            result.raise_for_status()
            return result.json()["candidates"][0]["content"]["parts"][0]["text"]

        elif self.provider == "anthropic":
            if not Anthropic:
                raise ImportError("Install anthropic: pip install anthropic")
            client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model=self.model,
                max_tokens=5000,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            )
            return response.content[0].text

        elif self.provider == "openai":
            if not OpenAI:
                raise ImportError("Install openai: pip install openai")
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                max_tokens=5000
            )
            return response.choices[0].message.content

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    # ------------------------------------------------------------------
    # Dependency management
    # ------------------------------------------------------------------
    def parse_imports(self, code: str) -> List[str]:
        """Extract top-level imports from code."""
        try:
            tree = ast.parse(code)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(name.name.split('.')[0] for name in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module.split('.')[0])
            return sorted(set(imports))
        except SyntaxError:
            return []

    def install_missing_packages(self, packages: List[str]):
        installed = {pkg.key.lower() for pkg in pkg_resources.working_set}
        to_install = [p for p in packages if p.lower() not in installed]
        if not to_install:
            return 0, "", ""
        process = subprocess.run([sys.executable, "-m", "pip", "install"] + to_install,
                                 capture_output=True, text=True)
        return process.returncode, process.stdout, process.stderr

    # ------------------------------------------------------------------
    # Code extraction & execution
    # ------------------------------------------------------------------
    def response_to_pycode(self, response: str) -> Optional[str]:
        """Extract ```python code``` blocks."""
        match = re.search(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip() if response.strip() else None

    def run_script_realtime(self, filepath: str = "./outputs/pycode.py"):
        process = subprocess.Popen([sys.executable, filepath],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True, bufsize=1)
        stdout_lines, stderr_lines = [], []

        def stream(pipe, lines, prefix=""):
            for line in iter(pipe.readline, ''):
                lines.append(line)
                print(prefix + line, end="")
            pipe.close()

        t1 = threading.Thread(target=stream, args=(process.stdout, stdout_lines))
        t2 = threading.Thread(target=stream, args=(process.stderr, stderr_lines, "[stderr] "))
        t1.start(); t2.start()
        process.wait(); t1.join(); t2.join()
        return process.returncode, ''.join(stdout_lines), ''.join(stderr_lines)

    # ------------------------------------------------------------------
    # Orchestration: Text → Code → Install → Debug → Run
    # ------------------------------------------------------------------
    def __call__(self, prompt: str):
        """End-to-end pipeline: prompt → code → install → run → debug if fails."""
        response = self.generate(prompt)
        pycode = self.response_to_pycode(response)
        if not pycode:
            raise ValueError("No Python code returned.")

        # Save file
        with open("./outputs/pycode.py", "w", encoding="utf-8") as f:
            f.write(pycode)

        # Install packages
        imports = self.parse_imports(pycode)
        self.install_missing_packages(imports)

        # Run script with retry debugging
        ret, out, err = self.run_script_realtime()
        attempt = 0
        while ret != 0 and attempt < self.attempt_limit:
            attempt += 1
            debug_prompt = (
                f"Fix the following Python code.\nError:\n{err}\n\nCode:\n{pycode}\n\n"
                "Return only full valid Python code in a code block."
            )
            response = self.generate(debug_prompt)
            pycode = self.response_to_pycode(response)
            if not pycode:
                break
            with open("./outputs/pycode.py", "w", encoding="utf-8") as f:
                f.write(pycode)
            imports = self.parse_imports(pycode)
            self.install_missing_packages(imports)
            ret, out, err = self.run_script_realtime()

        return ret, out, err
