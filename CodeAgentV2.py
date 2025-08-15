import os
import sys
import re
import ast
import threading
import subprocess
import requests
import pkg_resources
import importlib.util
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

    # ---------------- LLM Interaction ----------------
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
        elif self.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={os.environ.get('GEMINI_API_KEY')}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            headers = {"Content-Type": "application/json"}
        else:
            raise ValueError("Unknown provider. Use 'perplexity' or 'gemini'.")

        result = requests.post(url, json=payload, headers=headers)
        result.raise_for_status()
        data = result.json()

        if self.provider == "perplexity":
            return data["choices"][0]["message"]["content"]
        else:
            return data["candidates"][0]["content"]["parts"][0]["text"]

    # ---------------- Code Extraction ----------------
    def response_to_pycode(self, response: str) -> Optional[str]:
        """Extract Python code block from LLM response."""
        code_block_pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
        match = code_block_pattern.search(response)
        if match:
            return match.group(1).strip()
        if response.strip() and not response.strip().startswith("{"):
            return response.strip()
        return None

    def response_to_pyfile(self, response: str, filename: str = "./outputs/pycode.py"):
        pycode = self.response_to_pycode(response)
        if pycode is None:
            raise ValueError("No Python code block found in response.")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(pycode)
        print(f"[INFO] Saved extracted Python code to {filename}")

    # ---------------- Dependency Handling ----------------
    @staticmethod
    def is_stdlib_package(package: str) -> bool:
        """Check if a package is part of the Python standard library."""
        package = package.split('[')[0]
        if package in sys.builtin_module_names:
            return True
        spec = importlib.util.find_spec(package)
        if spec and spec.origin:
            return "site-packages" not in spec.origin
        return False

    @staticmethod
    def parse_imports(code: str):
        """Parse imports using AST for accurate detection."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            print("[WARNING] Invalid Python code for AST parsing. Falling back to regex.")
            return CodeAgent._parse_imports_regex(code)

        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(name.name.split('.')[0] for name in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split('.')[0])
        return sorted(imports)

    @staticmethod
    def _parse_imports_regex(code: str):
        """Fallback regex parsing for invalid code."""
        imports = set()
        for line in code.splitlines():
            line = line.strip()
            if line.startswith(("import ", "from ")):
                match = re.match(r"(?:import|from)\s+([a-zA-Z0-9_]+)", line)
                if match:
                    imports.add(match.group(1))
        return sorted(imports)

    @staticmethod
    def extract_requirements_from_code(code: str):
        """Extract requirements from # Requirements: comment."""
        for line in code.splitlines():
            line = line.strip()
            if line.lower().startswith("# requirements:"):
                req_str = line.split(":", 1)[1].strip()
                req_str = re.sub(r'\s*\(.*?\)', '', req_str)
                reqs = [r for r in req_str.split() if r.lower() not in ['pip', 'install']]
                return reqs
        return []

    def get_package_version(self, package: str) -> Optional[str]:
        try:
            return pkg_resources.get_distribution(package).version
        except pkg_resources.DistributionNotFound:
            return None

    def generate_requirements(self, packages: list, filename: str = "./outputs/requirements.txt"):
        """Generate/update requirements.txt without stdlib packages."""
        if not packages:
            return

        existing_lines = []
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not any(x in line.lower() for x in ["none", "nan", "-"]):
                        existing_lines.append(line)

        requirements = []
        for package in packages:
            pkg_name = package.split('==')[0]
            if '[' in pkg_name:
                pkg_name = pkg_name.split('[')[0]

            if self.is_stdlib_package(pkg_name):
                continue

            if '==' in package:
                requirements.append(package)
                continue

            version = self.get_package_version(pkg_name)
            if pkg_name == "sphinx" and version:
                try:
                    major_version = float(version.split('.')[0])
                    if not (5.1 <= major_version < 6.0):
                        requirements.append("sphinx==5.3.0")
                        continue
                except ValueError:
                    pass
            if version:
                requirements.append(f"{pkg_name}=={version}")
            else:
                requirements.append(package)

        all_requirements = sorted(set(existing_lines + requirements))
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(all_requirements) + "\n")
        print(f"[INFO] Updated requirements.txt at {filename}")

    def install_missing_packages(self, packages):
        """Install only non-stdlib, missing packages."""
        installed = {pkg.key.lower() for pkg in pkg_resources.working_set}
        to_install = []
        for p in packages:
            pkg_name = p.split('==')[0].lower()
            if '[' in pkg_name:
                pkg_name = pkg_name.split('[')[0]
            if self.is_stdlib_package(pkg_name):
                continue
            if pkg_name not in installed:
                to_install.append(p)

        if not to_install:
            print("[INFO] All dependencies already installed.")
            return 0, "", ""

        print(f"[INFO] Installing missing packages: {to_install}")
        process = subprocess.run([sys.executable, "-m", "pip", "install"] + to_install,
                                 capture_output=True, text=True)
        print(process.stdout)
        if process.stderr:
            print(process.stderr)
        return process.returncode, process.stdout, process.stderr

    def dependency_manager(self, code):
        """Manage dependencies and generate requirements.txt."""
        packages = self.extract_requirements_from_code(code) or self.parse_imports(code)
        if not packages:
            print("[INFO] No dependencies found — skipping install.")
            return 0, "", ""

        self.generate_requirements(packages)
        return self.install_missing_packages(packages)

    # ---------------- Script Execution ----------------
    def run_script_realtime(self, filepath: str = "./outputs/pycode.py"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found.")

        process = subprocess.Popen([sys.executable, filepath],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, bufsize=1)

        stdout_lines, stderr_lines = [], []

        def stream_output(pipe, lines_list, is_err=False):
            for line in iter(pipe.readline, ''):
                lines_list.append(line)
                print(f"[stderr] {line}" if is_err else line, end='')
            pipe.close()

        t1 = threading.Thread(target=stream_output, args=(process.stdout, stdout_lines, False))
        t2 = threading.Thread(target=stream_output, args=(process.stderr, stderr_lines, True))
        t1.start(), t2.start()
        process.wait()
        t1.join(), t2.join()

        return process.returncode, ''.join(stdout_lines), ''.join(stderr_lines)

    # ---------------- Debug Loop ----------------
    def __call__(self, prompt: str):
        prompt_with_rule = prompt.strip() + "\n\nIMPORTANT: Return ONLY a valid Python code block. At the top of the code, include a single line comment with the pip-installable packages if any are needed: # Requirements: package1 package2 ..."
        response = self.generate(prompt_with_rule)
        print("[INFO] Generated response (truncated):")
        print(response[:500] + ("..." if len(response) > 500 else ""))

        self.response_to_pyfile(response)
        with open("./outputs/pycode.py", "r", encoding="utf-8") as f:
            code = f.read()

        install_ret, install_out, install_err = self.dependency_manager(code)
        failed_on_install = install_ret != 0
        if not failed_on_install:
            retcode, out, err = self.run_script_realtime()
        else:
            retcode, out, err = install_ret, install_out, install_err

        attempt = 0
        while retcode != 0 and attempt < 10:
            attempt += 1
            print(f"[ERROR] {'Install' if failed_on_install else 'Run'} failed, attempt {attempt}...")

            debug_prompt = (
                f"You are a Debugger. Fix the Python code based on this error.\n"
                f"Error:\n{err}\n\nSource Code:\n{code}\n"
                f"Original Prompt: {prompt}\n"
                "IMPORTANT: Return ONLY a valid Python code block and it should be the FULL ENTIRE CODE."
            )
            debug_response = self.generate(debug_prompt)
            pycode = self.response_to_pycode(debug_response)
            if not pycode:
                print("[ERROR] No valid Python code returned — stopping.")
                break

            with open("./outputs/pycode.py", "w", encoding="utf-8") as f:
                f.write(pycode)

            install_ret, install_out, install_err = self.dependency_manager(pycode)
            failed_on_install = install_ret != 0
            if failed_on_install:
                retcode, out, err = install_ret, install_out, install_err
            else:
                retcode, out, err = self.run_script_realtime()

        if retcode == 0:
            print("[INFO] Script executed successfully!")
            self.generate_requirements(self.extract_requirements_from_code(code) or self.parse_imports(code))
        else:
            print(f"[ERROR] Script still failed after {attempt} attempts.")

        return retcode, out, err
