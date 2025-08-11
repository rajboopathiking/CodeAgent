import os
import requests
import subprocess

class CodeAgent:
    SUPPORTED_APIS = {
        "perplexity": {
            "url": "https://api.perplexity.ai/chat/completions",
            "key_env": "PPLX_API_KEY",
            "default_model": "sonar-pro"
        },
        "openai": {
            "url": "https://api.openai.com/v1/chat/completions",
            "key_env": "OPENAI_API_KEY",
            "default_model": "gpt-4o-mini"
        },
        "anthropic": {
            "url": "https://api.anthropic.com/v1/messages",
            "key_env": "ANTHROPIC_API_KEY",
            "default_model": "claude-3-opus-20240229"
        },
        "gemini": {
            "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            "key_env": "GEMINI_API_KEY",
            "default_model": "gemini-pro"
        },
        "grok": {
            "url": "https://api.x.ai/v1/chat/completions",
            "key_env": "XAI_API_KEY",
            "default_model": "grok-1"
        },
        "groq": {
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "key_env": "GROQ_API_KEY",
            "default_model": "mixtral-8x7b-32768"
        }
    }

    def __init__(self, apikey, provider="perplexity", model=None):
        if provider not in self.SUPPORTED_APIS:
            raise ValueError(f"Provider '{provider}' not supported. Choose from: {list(self.SUPPORTED_APIS.keys())}")
        self.provider = provider
        self.model = model or self.SUPPORTED_APIS[provider]["default_model"]
        self.prompt = None
        key_env_var = self.SUPPORTED_APIS[provider]["key_env"]
        if not os.environ.get(key_env_var):
            os.environ[key_env_var] = apikey

    def set_provider(self, provider, model=None):
        """Switch to a different API provider and optionally change the model."""
        if provider not in self.SUPPORTED_APIS:
            raise ValueError(f"Provider '{provider}' not supported.")
        self.provider = provider
        self.model = model or self.SUPPORTED_APIS[provider]["default_model"]

    def generate(self, prompt, retries=2, fallback=True):
        """Generate a response with retry and fallback support."""
        api_info = self.SUPPORTED_APIS[self.provider]
        url = api_info["url"]
        model = self.model
        headers = {"Content-Type": "application/json"}
        if self.provider != "gemini":
            headers["Authorization"] = f"Bearer {os.environ.get(api_info['key_env'])}"

        # Provider-specific payload
        if self.provider in ("perplexity", "openai", "grok", "groq"):
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2000}
        elif self.provider == "anthropic":
            payload = {"model": model, "max_tokens": 1024, "messages": [{"role": "user", "content": prompt}]}
        elif self.provider == "gemini":
            url = url.format(model=model) + f"?key={os.environ.get(api_info['key_env'])}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}

        # Retry loop
        for attempt in range(retries + 1):
            try:
                result = requests.post(url, json=payload, headers=headers, timeout=30)
                if result.status_code == 200:
                    response = result.json()
                    os.makedirs("./outputs/", exist_ok=True)
                    with open("./outputs/response_1.txt", "w") as file:
                        file.write(str(response))
                    print(f"[INFO] Response received from {self.provider} (model: {model})")
                    return result
                else:
                    print(f"[WARN] {self.provider} failed (status {result.status_code}). Attempt {attempt+1}/{retries}")
            except requests.RequestException as e:
                print(f"[ERROR] {self.provider} request failed: {e}. Attempt {attempt+1}/{retries}")

        # Fallback to next provider
        if fallback:
            fallback_order = list(self.SUPPORTED_APIS.keys())
            current_idx = fallback_order.index(self.provider)
            for next_provider in fallback_order[current_idx+1:] + fallback_order[:current_idx]:
                print(f"[INFO] Falling back to provider: {next_provider}")
                self.set_provider(next_provider)
                return self.generate(prompt, retries=retries, fallback=False)

        raise RuntimeError(f"All retries and fallbacks failed for prompt: {prompt}")

    def response_to_pycode(self, response):
        if not isinstance(response, str):
            return None
        if not response.strip():
            return None
        start_marker = "```python"
        start_index = response.find(start_marker)
        if start_index == -1:
            return None
        start_index += len(start_marker)
        end_marker = "```"
        end_index = response.find(end_marker, start_index)
        if end_index == -1:
            code = response[start_index:].strip()
        else:
            code = response[start_index:end_index].strip()
        notes_markers = ["# ---- Notes ----", "# Notes", "## Notes", "---- Notes ----"]
        for marker in notes_markers:
            idx = code.find(marker)
            if idx != -1:
                code = code[:idx].strip()
        return code if code else None

    def response_to_pyfile(self, response):
        if isinstance(response, dict) or hasattr(response, "json"):
            try:
                response = response.json()
            except:
                pass
        if isinstance(response, dict):
            if self.provider == "gemini":
                content = response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            elif self.provider == "anthropic":
                content = response.get("content", [{}])[0].get("text", "")
            else:
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            content = str(response)
        code = self.response_to_pycode(content)
        os.makedirs("./outputs/", exist_ok=True)
        with open("./outputs/pycode.py", "w") as file:
            file.write(code or "")
        print("Files Created")

    def Workflow(self, prompt=None):
        process = subprocess.Popen(
            ["python", "./outputs/pycode.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        with open("./outputs/pycode.py", "r") as file:
            source_code = file.read()
        stdout_output = []
        stderr_output = []
        while True:
            stdout_line = process.stdout.readline()
            if stdout_line:
                print(f"[STDOUT] {stdout_line.strip()}")
                stdout_output.append(stdout_line)
            stderr_line = process.stderr.readline()
            if stderr_line:
                print(f"[STDERR] {stderr_line.strip()}")
                stderr_output.append(stderr_line)
            if process.poll() is not None:
                break
        for stdout_line in process.stdout:
            print(f"[STDOUT] {stdout_line.strip()}")
            stdout_output.append(stdout_line)
        for stderr_line in process.stderr:
            print(f"[STDERR] {stderr_line.strip()}")
            stderr_output.append(stderr_line)
        process.wait()
        print(f"Process finished with return code: {process.returncode}")
        if process.returncode != 0:
            print("Debugging...")
            debug_prompt = (
                f"You are Debugger. Fix the code based on this error.\n"
                f"Output: {''.join(stdout_output)}\n"
                f"Errors: {''.join(stderr_output)}\n"
                f"Source Code:\n{source_code}"
            )
            if prompt:
                debug_prompt += f"\nOriginal Prompt: {prompt}"
            response = self.generate(debug_prompt)
            self.response_to_pyfile(response)
            return process.returncode
        else:
            return process.returncode

    def __call__(self, prompt):
        self.prompt = prompt
        response = self.generate(prompt)
        self.response_to_pyfile(response)
        return self.Workflow()
