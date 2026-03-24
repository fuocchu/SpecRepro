"""
BaseAgent: shared infrastructure for all SpecRepro agents.

Maintains a rolling conversation history and delegates LLM calls
to utils.llm.query_llm.
"""

from specrepro.utils.llm import query_llm


class BaseAgent:
    """
    Base class for all SpecRepro agents.

    Each agent maintains:
      - a rolling history of (phase, prompt_excerpt, response) tuples
      - a reference to the current PaperSpec
    """

    MAX_HISTORY = 6 

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        verbose: bool = True,
        print_cost: bool = True,
    ):
        self.model = model
        self.verbose = verbose
        self.print_cost = print_cost
        self.history: list[dict] = []   # {"phase": ..., "response": ...}

    def inference(
        self,
        prompt: str,
        phase: str,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> str:
        if self.verbose:
            print(f"  [{self.__class__.__name__}] phase={phase}")

        system = self.system_prompt()
        full_prompt = self._build_prompt(prompt, phase)

        response = query_llm(
            prompt=full_prompt,
            system_prompt=system,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            print_cost=self.print_cost,
        )
        self.history.append({"phase": phase, "response": response[:500]})
        if len(self.history) > self.MAX_HISTORY:
            self.history.pop(0)

        return response


    def system_prompt(self) -> str:
        return "You are a helpful AI research assistant."

    def _build_prompt(self, prompt: str, phase: str) -> str:
        """Default: pass prompt through unchanged. Subclasses may add context."""
        return prompt

    def reset(self):
        self.history.clear()
