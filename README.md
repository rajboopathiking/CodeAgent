🚀 CodeAgent  

Automate code generation, execution, and debugging for your projects using **LLM-powered agents**.  
Supports multiple providers (`Proplexity`, `Gemini`, and more), multimodal input, and dependency management.  

<p align="center">
  <a href="https://pypi.org/project/c4agent/"><img src="https://img.shields.io/pypi/v/c4agent?color=blue&label=PyPI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg"></a>
  <a href="https://github.com/rajboopathiking/CodeAgent/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg"></a>
  <a href="https://github.com/rajboopathiking/CodeAgent/"><img src="https://img.shields.io/badge/build-passing-brightgreen.svg"></a>
</p>

---

## 📦 Installation

Install from PyPI:

```bash
pip install c4agent
```

Or install from source:

```bash
git clone https://github.com/yourusername/CodeAgent.git
cd CodeAgent
pip install -r requirements.txt
```

## ⚡ Quick Start

### Initialize Agent

```python
from Agent.CodeAgent import CodeAgent        # v1
from Agent.CodeAgentV2 import CodeAgent      # v2
from Agent.CodeAgentV3 import CodeAgent      # v3

# Example: Initialize with Proplexity API
agent = CodeAgent("<apikey>")
```

## 🧑‍💻 Versions

### CodeAgent V1
- Generates & runs Python projects
- Provider: "proplexity"

### CodeAgent V2
- Generates & runs Python projects
- Providers: "proplexity", "gemini"
- Dependency Manager included

### CodeAgent V3
- Generates & runs Python projects
- Supports multiple providers
- Dependency Manager
- Multimodal input (Text + Images)

## ✨ Usage

### 🔹 1. Generate Code from Prompt

```python
agent.generate(
    "Explain About Artificial Intelligence"
).json()
```

### 🔹 2. Automate Flow - Example Project

```python
prompt = """
You are an AI Agent. You will code like an AI research scientist.

Code For SmolAgents

Instructions:
1. Agent should answer tech-related questions
2. Execution not supported
3. Give only Python code
4. Python only support
5. Should include docstrings

User: Build a multimodal embedding model (Image + Text) using contrastive learning.

Dataset Link and Description:
- Kaggle credentials are already set up
- Dataset: fashion-product-images-small

Load dataset:
```python
!mkdir -p /root/.kaggle
!cp kaggle.json /root/.kaggle
!chmod 600 /root/.kaggle/kaggle.json
!kaggle datasets download paramaggarwal/fashion-product-images-small
```

Dataset load using Python:
```python
import pandas as pd
df = pd.read_csv("/content/myntradataset/styles.csv", on_bad_lines="skip")
df.head()
```

Example dataset output:
```
   id    gender    masterCategory    subCategory    articleType    baseColour    season    year    usage    productDisplayName
0  15970  Men      Apparel         Topwear        Shirts        Navy Blue    Fall      2011.0  Casual   Turtle Check Men Navy Blue Shirt
1  39386  Men      Apparel         Bottomwear     Jeans         Blue         Summer   2012.0  Casual   Peter England Men Party Blue Jeans
2  59263  Women    Accessories     Watches        Watches       Silver       Winter   2016.0  Casual   Titan Women Silver Watch
3  21379  Men      Apparel         Bottomwear     Track Pants   Black        Fall      2011.0  Casual   Manchester United Men Solid Black Track Pants
4  53759  Men      Apparel         Topwear        Tshirts       Grey         Summer   2012.0  Casual   Puma Men Grey T-shirt
```

Model Requirements:
- Use HuggingFace pretrained BERT and ViT models
- Train using contrastive learning
- Use Torch and optionally LangChain
- Save best model & logs
- Include evaluation, testing, and CUDA support
- Progress bar using tqdm
- Provide full final code
"""

# Run the agent
agent(prompt)
```

### 🔹 3. V3 Multimodal Example

```python
agent = CodeAgent(
    gemini_apikey="<apikey>",
    provider="gemini"
)

result = agent({
    "text": "Write a Python script to save a plot in ./plot.png",
    "images": ["/content/Loss.png", "/content/Accuracy.png"]
})

print(result)
```

📂 Outputs are stored in local folders.

## 📑 Example Output

When running prompts, CodeAgent will:
- ✅ Generate full Python code
- ✅ Manage dependencies
- ✅ Save outputs & logs locally
- ✅ Handle debugging & execution automatically

## 🔧 Requirements
- Python 3.8+
- Dependencies (auto-installed with `pip install c4agent`)

## 📌 Roadmap
- Support Proplexity provider
- Add Gemini provider
- Dependency manager
- Multimodal input (text + images)
- Add more providers (OpenAI, Claude, etc.)
- CLI support
- Web UI for interactive coding

## 🤝 Contributing
Contributions are welcome!
1. Fork the repo
2. Create your feature branch (`git checkout -b feature/awesome-feature`)
3. Commit changes (`git commit -m 'Add awesome feature'`)
4. Push to branch (`git push origin feature/awesome-feature`)
5. Open a Pull Request

## 📜 License
MIT License © 2025

## 🌟 Support
If you like this project, please ⭐ the repo to support development!


______________________________________________________________________________

V3 API Documentation

```python
class CodeAgent:
    """
    A multimodal code generation and execution agent.

    CodeAgent interacts with multiple LLM providers (Perplexity, Gemini, Anthropic, OpenAI),
    handles multimodal inputs (text, images, PDFs), generates Python code, installs missing
    dependencies, and executes code with iterative debugging.

    :param pplx_apikey: Perplexity API key.
    :type pplx_apikey: str, optional
    :param gemini_apikey: Gemini API key.
    :type gemini_apikey: str, optional
    :param anthropic_apikey: Anthropic API key.
    :type anthropic_apikey: str, optional
    :param openai_apikey: OpenAI API key.
    :type openai_apikey: str, optional
    :param provider: Model provider, one of {"perplexity", "gemini", "anthropic", "openai"}.
    :type provider: str, default="perplexity"
    :param model: Model identifier; if None, defaults to provider’s default.
    :type model: str, optional
    """

    def generate(self, input_data: Union[str, dict]) -> str:
        """
        Generate a response from the configured provider.

        :param input_data: Either a plain prompt (str) or a dict containing
            ``{"text": str, "images": [paths], "pdfs": [paths]}``.
        :type input_data: str or dict
        :return: Model output as string.
        :rtype: str
        :raises ValueError: If provider is unknown.
        :raises requests.HTTPError: If API request fails.
        """
        ...

    def process_multimodal_input(self, input_data: Union[str, dict]) -> dict:
        """
        Process multimodal inputs into normalized format.

        :param input_data: Input string or dict with keys ``text``, ``images``, ``pdfs``.
        :type input_data: str or dict
        :return: Dictionary with keys ``text``, ``images``, ``files``.
        :rtype: dict
        :raises ValueError: If input is neither string nor dict.
        """
        ...

    def is_stdlib_package(self, package: str) -> bool:
        """
        Check if a package is part of the Python standard library.

        :param package: Package name.
        :type package: str
        :return: True if stdlib, False otherwise.
        :rtype: bool
        """
        ...

    def parse_imports(self, code: str) -> List[str]:
        """
        Extract imports from code using AST.

        :param code: Python source code.
        :type code: str
        :return: List of imported top-level modules.
        :rtype: list[str]
        """
        ...

    def extract_requirements_from_code(self, code: str) -> List[str]:
        """
        Extract requirements from a ``# Requirements:`` comment.

        :param code: Python source code.
        :type code: str
        :return: List of requirement strings.
        :rtype: list[str]
        """
        ...

    def generate_requirements(self, packages: list, filename: str = "./outputs/requirements.txt"):
        """
        Generate or update requirements.txt with detected dependencies.

        :param packages: List of package names.
        :type packages: list[str]
        :param filename: Path to requirements file.
        :type filename: str
        """
        ...

    def install_missing_packages(self, packages: List[str]) -> tuple[int, str, str]:
        """
        Install missing packages via pip.

        :param packages: List of package names with optional version specifiers.
        :type packages: list[str]
        :return: (return_code, stdout, stderr)
        :rtype: tuple[int, str, str]
        """
        ...

    def dependency_manager(self, code: str) -> tuple[int, str, str]:
        """
        Detect and install dependencies based on code.

        - Parses imports and requirements.
        - Adds provider-specific deps (anthropic, openai).
        - Adds Pillow/PyPDF2 if handling images/PDFs.

        :param code: Python source code.
        :type code: str
        :return: (return_code, stdout, stderr)
        :rtype: tuple[int, str, str]
        """
        ...

    def response_to_pycode(self, response: str) -> Optional[str]:
        """
        Extract Python code from model response (inside ```python ...```).

        :param response: Model response text.
        :type response: str
        :return: Extracted Python code or None.
        :rtype: str or None
        """
        ...

    def response_to_pyfile(self, response: str, filename: str = "./outputs/pycode.py"):
        """
        Save extracted Python code to file.

        :param response: Model response text.
        :type response: str
        :param filename: Path to output file.
        :type filename: str
        :raises ValueError: If no Python code block found.
        """
        ...

    def run_script_realtime(self, filepath: str = "./outputs/pycode.py") -> tuple[int, str, str]:
        """
        Run Python script with real-time stdout/stderr capture.

        :param filepath: Path to script.
        :type filepath: str
        :return: (return_code, stdout, stderr)
        :rtype: tuple[int, str, str]
        :raises FileNotFoundError: If file not found.
        """
        ...

    def __call__(self, input_data: Union[str, dict]) -> tuple[int, str, str]:
        """
        Generate, install dependencies, run, and debug Python code.

        - Ensures output is a valid Python code block with ``# Requirements:``.
        - Saves to file and installs dependencies.
        - Runs script and retries debugging up to 10 times if it fails.
        - Updates requirements.txt on success.

        :param input_data: Prompt string or dict with multimodal input.
        :type input_data: str or dict
        :return: (exit_code, stdout, stderr)
        :rtype: tuple[int, str, str]
        """
        ...

```
