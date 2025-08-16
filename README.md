# 🚀 CodeAgent  

Automate code generation, execution, and debugging for your projects using **LLM-powered agents**.  
Supports multiple providers (`Proplexity`, `Gemini`, and more), multimodal input, and dependency management.  

<p align="center">
  <a href="https://pypi.org/project/c4agent/"><img src="https://img.shields.io/pypi/v/c4agent?color=blue&label=PyPI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg"></a>
  <img src="https://img.shields.io/badge/license-MIT-green.svg">
  <img src="https://img.shields.io/badge/build-passing-brightgreen.svg">
</p>

---

## 📦 Installation  

Install from **PyPI**:

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
4. Docstrings required

User: Build a multimodal embedding model (Image + Text) using contrastive learning.

Dataset:
- Kaggle credentials are already set up
- Dataset: fashion-product-images-small

Example dataset load:
```python
import pandas as pd
df = pd.read_csv("/content/myntradataset/styles.csv", on_bad_lines="skip")
df.head()
```

Model Requirements:
- Use HuggingFace pretrained BERT and ViT models
- Train using contrastive learning
- Use Torch and optionally LangChain
- Save best model & logs
- Include evaluation, testing, and CUDA support
- Progress bar using tqdm
- Give full final code.
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
    "images": ["/content/Loss.png"/content/Accuracy.png"]
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
