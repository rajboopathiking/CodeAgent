import os
import sys
import io
import contextlib
import requests
import traceback
from typing import Optional, List

class CodeAgent:
    def __init__(self, apikey: str):
        self.apikey = apikey
        if not os.environ.get("PPLX_API_KEY"):
            os.environ["PPLX_API_KEY"] = self.apikey
        os.makedirs("./outputs/", exist_ok=True)

    def generate(self, prompt: str) -> str:
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

    def split_into_cells(self, code: str) -> List[str]:
      raw_cells = code.split("\n\n")  # split on double newline
      merged_cells = []
      buffer = ""
      for raw in raw_cells:
          buffer += ("\n\n" + raw) if buffer else raw
          if (buffer.count('"""') % 2 == 0) and (buffer.count("'''") % 2 == 0):
              merged_cells.append(buffer.strip())
              buffer = ""
      if buffer:  # leftover if still unbalanced
          merged_cells.append(buffer.strip())
      return merged_cells


    def run_cells(self, code: str):
        """Run code cell-by-cell, showing outputs."""
        namespace = {}
        cells = self.split_into_cells(code)

        for idx, cell in enumerate(cells, start=1):
            print(f"\n📦 Executing Cell {idx}:\n{'-'*40}")
            print(cell)
            print("-"*40)

            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()

            try:
                with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                    exec(cell, namespace)
            except Exception as e:
                print(stdout_buffer.getvalue(), end="")
                print(stderr_buffer.getvalue(), end="")
                print(f"❌ Error in Cell {idx}: {e}")
                traceback.print_exc()
                return False, idx, stdout_buffer.getvalue(), stderr_buffer.getvalue(), str(e)

            print(stdout_buffer.getvalue(), end="")
            print(stderr_buffer.getvalue(), end="")

        return True, None, None, None, None

    def Workflow(self, prompt: str, max_retries: int = 10):
        """Generate → run cell-by-cell → fix failing cell → retry."""
        print("🚀 Generating initial code...")
        initial_code = self.response_to_pycode(self.generate(prompt))
        if not initial_code:
            raise ValueError("❌ LLM did not return valid Python code.")

        for attempt in range(max_retries):
            print("\n" + "="*60)
            print(f"🌀 Attempt {attempt + 1}")
            print("="*60)

            success, fail_cell_idx, stdout_data, stderr_data, err_msg = self.run_cells(initial_code)

            if success:
                print("✅ All cells executed successfully.")
                return 0

            print(f"🔄 Debugging cell {fail_cell_idx} with LLM...")
            debug_prompt = (
                f"You are a Python debugger.\n"
                f"Original task: {prompt}\n"
                f"The following cell caused an error:\n"
                f"{self.split_into_cells(initial_code)[fail_cell_idx-1]}\n"
                f"STDOUT:\n{stdout_data}\n"
                f"STDERR:\n{stderr_data}\n"
                f"Error message: {err_msg}\n"
                f"Please return the FULL corrected Python script."
            )

            new_code = self.response_to_pycode(self.generate(debug_prompt))
            if not new_code or new_code.strip() == initial_code.strip():
                print("⚠️ Code unchanged after fix attempt. Stopping.")
                break

            initial_code = new_code  # retry with updated code

        print("❌ Max retries reached without success.")
        return 1