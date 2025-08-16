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
