class CodeAgent:
  def __init__(self,apikey):
    import os

    self.apikey = apikey
    self.prompt = None

    if not os.environ.get("PPLX_API_KEY"):
      os.environ["PPLX_API_KEY"] = self.apikey

  def generate(self,prompt):
    import requests
    import os
    import getpass
    from IPython.display import display

    url = "https://api.perplexity.ai/chat/completions"

    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 50000
    }
    headers = {
        "Authorization": f"Bearer {os.environ.get('PPLX_API_KEY')}",
        "Content-Type": "application/json"
    }

    result = requests.post(url, json=payload, headers=headers)
    response = result.json()

    import os
    os.makedirs("./outputs/",exist_ok=True)
    with open("./outputs/response_1.txt","w") as file:
      file.writelines(response["choices"][0]["message"]["content"])

    return result

  def response_to_pycode(self,response):
    """
    Extract Python code from a response string containing a code block, removing any trailing notes or descriptions.

    Args:
        response: A string containing a potential Python code block marked by triple backticks (```python).

    Returns:
        str: The extracted Python code as a string, stripped of leading/trailing whitespace and trailing notes.
        None: If the input is invalid, no valid Python code block is found, or the block is empty.

    Examples:
        >>> response = "Some text\\n```python\\nprint('hello')\\n```\\nMore text"
        >>> response_to_pycode(response)
        "print('hello')"
        >>> response_to_pycode("No code here")
        None
        >>> response = "```python\\nprint('test')\\n```\\n# ---- Notes ----\\n# Example note"
        >>> response_to_pycode(response)
        "print('test')"
    """
    # Validate input type
    if not isinstance(response, str):
        print(f"Error: Invalid input type. Expected str, got {type(response).__name__}.")
        return None

    # Check if response is empty or only whitespace
    if not response or response.isspace():
        print("Error: Input string is empty or contains only whitespace.")
        return None

    # Find the start of the Python code block
    start_marker = "```python"
    start_index = response.find(start_marker)
    if start_index == -1:
        print("Warning: No Python code block found (missing '```python').")
        return None

    # Adjust start_index to point to the actual code content
    start_index += len(start_marker)

    # Find the end of the code block by looking for the first standalone "```"
    end_marker = "```"
    end_index = -1
    current_index = start_index
    while current_index < len(response):
        next_backtick = response.find(end_marker, current_index)
        if next_backtick == -1:
            print("Warning: Closing '```' not found after '```python'. Extracting until end of string.")
            return response[start_index:].strip()

        # Check if the backticks are standalone (not part of another code block or text)
        # Ensure it's not immediately followed by "python" or other text that indicates a new code block
        if next_backtick + len(end_marker) >= len(response) or \
           response[next_backtick + len(end_marker):].lstrip().startswith(('\n', ' ', '\t', '\r')) or \
           response[next_backtick + len(end_marker)] in ('\n', ' ', '\t', '\r', ''):
            end_index = next_backtick
            break

        # Move past this backtick to find the next one
        current_index = next_backtick + len(end_marker)

    if end_index == -1:
        print("Warning: No valid closing '```' found. Extracting until end of string.")
        return response[start_index:].strip()

    # Extract the code block and remove trailing notes
    code = response[start_index:end_index].strip()
    if not code:
        print("Warning: Python code block is empty.")
        return None

    # Explicitly remove any trailing notes section (e.g., "# ---- Notes ----")
    # This is an additional safeguard, though the end_index should already exclude notes
    notes_markers = ["# ---- Notes ----", "# Notes", "## Notes", "---- Notes ----"]
    for marker in notes_markers:
        notes_index = code.find(marker)
        if notes_index != -1:
            code = code[:notes_index].strip()

    # Final check to ensure code is not empty after removing notes
    if not code:
        print("Warning: Code block is empty after removing notes.")
        return None

    return code

  def response_to_pyfile(self,response):
    response = response.json()["choices"][0]["message"]["content"]
    code = self.response_to_pycode(response)
    import os
    os.makedirs("./outputs/",exist_ok=True)
    with open("./outputs/pycode.py","w") as file:
      file.write(code)

    print("Files Created")

  def Workflow(self,prompt=None):
    from sys import stdout
    import subprocess

    # Run the pycode.py script and stream logs in real-time
    process = subprocess.Popen(
        ["python", "./outputs/pycode.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # Line-buffered
    )

    with open("./outputs/pycode.py","r") as file:
      source_code = file.read()
    # Stream stdout and stderr in real-time
    stdout_output = []
    stderr_output = []

    while True:
        # Read stdout line by line
        stdout_line = process.stdout.readline()
        if stdout_line:
            print(f"[STDOUT] {stdout_line.strip()}")
            stdout_output.append(stdout_line)

        # Read stderr line by line
        stderr_line = process.stderr.readline()
        if stderr_line:
            print(f"[STDERR] {stderr_line.strip()}")
            stderr_output.append(stderr_line)

        # Check if the process has finished
        if process.poll() is not None:
            break

    # Capture any remaining output
    for stdout_line in process.stdout:
        print(f"[STDOUT] {stdout_line.strip()}")
        stdout_output.append(stdout_line)
    for stderr_line in process.stderr:
        print(f"[STDERR] {stderr_line.strip()}")
        stderr_output.append(stderr_line)


    # Ensure the process is complete
    process.wait()

    # Print the return code
    print(f"Process finished with return code: {process.returncode}")


    if process.returncode != 0:
      print("Debugging...")
      if prompt is None:
        prompt = f'You are Debugger , I will attach both code and error and output based on that i want corrected code : {"".join(stdout_output)} and {"".join(stderr_output)} and Source Code : {source_code} all code in together full final fixed corrected code'
      else:
        prompt = f'You are Debugger , I will attach both code and error and output based on that i want corrected code : {"".join(stdout_output)} and {"".join(stderr_output)} and Source Code : {source_code} all code in together full final fixed corrected code. Actual Prompt : {prompt}'
      response = self.generate(prompt)
      self.response_to_pyfile(response)
      return process.returncode
    else:
      return process.returncode

  def __call__(self,prompt):
    self.prompt = prompt


    response = self.generate(prompt)
    self.response_to_pyfile(response)
    return self.Workflow()
