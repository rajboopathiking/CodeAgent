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
        """Switch API provider and optionally change the model."""
        if provider not in self.SUPPORTED_APIS:
            raise ValueError(f"Provider '{provider}' not supported.")
        self.provider = provider
        self.model = model or self.SUPPORTED_APIS[provider]["default_model"]

    def generate(self, prompt, retries=2, fallback=True):
        """Generate AI output with retry and fallback."""
        api_info = self.SUPPORTED_APIS[self.provider]
        url = api_info["url"]
        headers = {"Content-Type": "application/json"}
        if self.provider != "gemini":
            headers["Authorization"] = f"Bearer {os.environ.get(api_info['key_env'])}"

        if self.provider in ("perplexity", "openai", "grok", "groq"):
            payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2000}
        elif self.provider == "anthropic":
            payload = {"model": self.model, "max_tokens": 1024, "messages": [{"role": "user", "content": prompt}]}
        elif self.provider == "gemini":
            url = url.format(model=self.model) + f"?key={os.environ.get(api_info['key_env'])}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}

        for attempt in range(retries + 1):
            try:
                result = requests.post(url, json=payload, headers=headers, timeout=30)
                if result.status_code == 200:
                    response = result.json()
                    os.makedirs("./outputs/", exist_ok=True)
                    with open("./outputs/response_1.txt", "w") as file:
                        file.write(str(response))
                    print(f"[INFO] Response from {self.provider} ({self.model})")
                    return result
                else:
                    print(f"[WARN] {self.provider} failed (status {result.status_code}) attempt {attempt+1}")
            except requests.RequestException as e:
                print(f"[ERROR] {self.provider} request failed: {e} (attempt {attempt+1})")

        if fallback:
            fallback_list = list(self.SUPPORTED_APIS.keys())
            idx = fallback_list.index(self.provider)
            for next_provider in fallback_list[idx+1:] + fallback_list[:idx]:
                print(f"[INFO] Falling back to {next_provider}")
                self.set_provider(next_provider)
                return self.generate(prompt, retries=retries, fallback=False)

        raise RuntimeError("All providers failed.")

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
        for marker in ["# ---- Notes ----", "# Notes", "## Notes", "---- Notes ----"]:
            idx = code.find(marker)
            if idx != -1:
                code = code[:idx].strip()
        return code if code else None

    def response_to_pyfile(self, response):
        if hasattr(response, "json"):
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
        stdout_output, stderr_output = [], []
        while True:
            out_line = process.stdout.readline()
            if out_line:
                print(f"[STDOUT] {out_line.strip()}")
                stdout_output.append(out_line)
            err_line = process.stderr.readline()
            if err_line:
                print(f"[STDERR] {err_line.strip()}")
                stderr_output.append(err_line)
            if process.poll() is not None:
                break
        for out_line in process.stdout:
            print(f"[STDOUT] {out_line.strip()}")
            stdout_output.append(out_line)
        for err_line in process.stderr:
            print(f"[STDERR] {err_line.strip()}")
            stderr_output.append(err_line)
        process.wait()
        print(f"Process finished with return code: {process.returncode}")
        if process.returncode != 0:
            print("Debugging...")
            if prompt is None:
                debug_prompt = (
                    f"You are Debugger. Fix the code.\nOutput: {''.join(stdout_output)}\n"
                    f"Errors: {''.join(stderr_output)}\nSource Code:\n{source_code}"
                )
            else:
                debug_prompt = (
                    f"You are Debugger. Fix the code.\nOutput: {''.join(stdout_output)}\n"
                    f"Errors: {''.join(stderr_output)}\nSource Code:\n{source_code}\nOriginal Prompt: {prompt}"
                )
            response = self.generate(debug_prompt)
            self.response_to_pyfile(response)
            return process.returncode
        return process.returncode

    def __call__(self, prompt):
        self.prompt = prompt
        response = self.generate(prompt)
        self.response_to_pyfile(response)
        return self.Workflow()
