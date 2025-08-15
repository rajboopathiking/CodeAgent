import os
import sys
import re
import ast
import threading
import subprocess
import requests
import pkg_resources
import importlib.util
from typing import Optional, Union, List
import base64
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

import sys
import re
import ast
import threading
import subprocess
import requests
import pkg_resources
import importlib.util
from typing import Optional, Union, List
import base64
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
    def __init__(self, pplx_apikey: Optional[str] = None, gemini_apikey: Optional[str] = None,
                 anthropic_apikey: Optional[str] = None, openai_apikey: Optional[str] = None,
                 provider: str = "perplexity", model: Optional[str] = None):
        self.provider = provider.lower()
        # Default models for each provider
        self.default_models = {
            "perplexity": "sonar-pro",
            "gemini": "gemini-1.5-flash",  # Updated default for vision support
            "anthropic": "claude-3-5-sonnet-20241022",
            "openai": "gpt-4o"
        }
        self.model = model or self.default_models.get(self.provider, "gemini-1.5-flash")
        self.pplx_apikey = pplx_apikey
        self.gemini_apikey = gemini_apikey
        self.anthropic_apikey = anthropic_apikey
        self.openai_apikey = openai_apikey

        if self.pplx_apikey and not os.environ.get("PPLX_API_KEY"):
            os.environ["PPLX_API_KEY"] = self.pplx_apikey
        if self.gemini_apikey and not os.environ.get("GEMINI_API_KEY"):
            os.environ["GEMINI_API_KEY"] = self.gemini_apikey
        if self.anthropic_apikey and not os.environ.get("ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_apikey
        if self.openai_apikey and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = self.openai_apikey

        os.makedirs("./outputs/", exist_ok=True)

    def generate(self, input_data: Union[str, dict]) -> str:
        processed_input = self.process_multimodal_input(input_data)
        prompt = processed_input["text"]
        images = processed_input["images"]
        files = processed_input["files"]

        for file in files:
            if file["type"] == "pdf":
                prompt += f"\n\n[Extracted from {file['path']}]\n{file['content']}"

        if self.provider == "perplexity":
            url = "https://api.perplexity.ai/chat/completions"
            messages = [{"role": "user", "content": prompt}]
            if images:
                print("[WARNING] Perplexity does not support image inputs. Ignoring images.")
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 50000
            }
            headers = {
                "Authorization": f"Bearer {os.environ.get('PPLX_API_KEY')}",
                "Content-Type": "application/json"
            }
            try:
                result = requests.post(url, json=payload, headers=headers)
                result.raise_for_status()
                return result.json()["choices"][0]["message"]["content"]
            except requests.HTTPError as e:
                print(f"[ERROR] Perplexity API call failed: {e}")
                raise

        elif self.provider == "gemini":
            # Updated endpoint for Gemini API
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={os.environ.get('GEMINI_API_KEY')}"
            parts = [{"text": prompt}]
            for img in images:
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img["data"]}})
            payload = {"contents": [{"parts": parts}]}
            headers = {"Content-Type": "application/json"}
            try:
                result = requests.post(url, json=payload, headers=headers)
                result.raise_for_status()
                return result.json()["candidates"][0]["content"]["parts"][0]["text"]
            except requests.HTTPError as e:
                if "404" in str(e):
                    print(f"[ERROR] Gemini model '{self.model}' not found. Retrying with default model 'gemini-1.5-flash'.")
                    self.model = "gemini-1.5-flash"
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={os.environ.get('GEMINI_API_KEY')}"
                    result = requests.post(url, json=payload, headers=headers)
                    result.raise_for_status()
                    return result.json()["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    print(f"[ERROR] Gemini API call failed: {e}")
                    raise

        elif self.provider == "anthropic":
            if not Anthropic:
                raise ImportError("Anthropic package not installed. Install with `pip install anthropic`.")
            client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            for img in images:
                messages[0]["content"].append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img["data"]
                    }
                })
            try:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=50000,
                    messages=messages
                )
                return response.content[0].text
            except Exception as e:
                print(f"[ERROR] Anthropic API call failed: {e}")
                raise

        elif self.provider == "openai":
            if not OpenAI:
                raise ImportError("OpenAI package not installed. Install with `pip install openai`.")
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            for img in images:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img['data']}"}
                })
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=50000
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[ERROR] OpenAI API call failed: {e}")
                raise

        else:
            raise ValueError("Unknown provider. Use 'perplexity', 'gemini', 'anthropic', or 'openai'.")

        os.makedirs("./outputs/", exist_ok=True)

    def process_multimodal_input(self, input_data: Union[str, dict]) -> dict:
        result = {"text": "", "images": [], "files": []}
        if isinstance(input_data, str):
            result["text"] = input_data
        elif isinstance(input_data, dict):
            result["text"] = input_data.get("text", "")
            for img_path in input_data.get("images", []):
                if os.path.exists(img_path):
                    with open(img_path, "rb") as img_file:
                        img_data = base64.b64encode(img_file.read()).decode("utf-8")
                        result["images"].append({"type": "image", "data": img_data, "path": img_path})
                else:
                    print(f"[WARNING] Image file {img_path} not found.")
            for pdf_path in input_data.get("pdfs", []):
                if os.path.exists(pdf_path):
                    try:
                        with open(pdf_path, "rb") as pdf_file:
                            reader = PyPDF2.PdfReader(pdf_file)
                            text = ""
                            for page in reader.pages:
                                text += page.extract_text() or ""
                            result["files"].append({"type": "pdf", "content": text, "path": pdf_path})
                    except Exception as e:
                        print(f"[ERROR] Failed to process PDF {pdf_path}: {e}")
                else:
                    print(f"[WARNING] PDF file {pdf_path} not found.")
        else:
            raise ValueError("Input must be a string or dict with text, images, and/or pdfs.")
        return result
        
    def is_stdlib_package(self, package: str) -> bool:
        package = package.split('[')[0]
        if package in sys.builtin_module_names:
            return True
        spec = importlib.util.find_spec(package)
        if spec and spec.origin:
            return "site-packages" not in spec.origin
        return False

    def parse_imports(self, code: str):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            print("[WARNING] Invalid Python code for AST parsing. Falling back to regex.")
            return self._parse_imports_regex(code)

        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(name.name.split('.')[0] for name in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split('.')[0])
        return sorted(imports)

    def _parse_imports_regex(self, code: str):
        imports = set()
        for line in code.splitlines():
            line = line.strip()
            if line.startswith(("import ", "from ")):
                match = re.match(r"(?:import|from)\s+([a-zA-Z0-9_]+)", line)
                if match:
                    imports.add(match.group(1))
        return sorted(imports)

    def extract_requirements_from_code(self, code: str):
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

    def dependency_manager(self, code: str):
        packages = self.extract_requirements_from_code(code) or self.parse_imports(code)
        additional_deps = []
        if self.provider == "anthropic":
            additional_deps.append("anthropic")
        if self.provider == "openai":
            additional_deps.append("openai")
        if "images" in str(code).lower() or "pdf" in str(code).lower():
            additional_deps.extend(["Pillow", "PyPDF2"])

        packages.extend(additional_deps)
        if not packages:
            print("[INFO] No dependencies found — skipping install.")
            return 0, "", ""

        self.generate_requirements(packages)
        return self.install_missing_packages(packages)

    def response_to_pycode(self, response: str) -> Optional[str]:
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

    def __call__(self, input_data: Union[str, dict]):
        prompt_with_rule = (
            input_data if isinstance(input_data, str)
            else input_data.get("text", "") +
            "\n\nIMPORTANT: Return ONLY a valid Python code block. At the top of the code, "
            "include a single line comment with the pip-installable packages if any are needed: "
            "# Requirements: package1 package2 ..."
        )
        input_data = input_data if isinstance(input_data, str) else {**input_data, "text": prompt_with_rule}
        response = self.generate(input_data)
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
                f"Original Input: {input_data}\n"
                "IMPORTANT: Return ONLY a valid Python code block and it should be the FULL ENTIRE CODE."
            )
            debug_response = self.generate({"text": debug_prompt})
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
