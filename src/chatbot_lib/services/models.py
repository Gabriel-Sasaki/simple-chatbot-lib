import abc
from typing import Any
from langchain_core.language_models import BaseChatModel

class ContextService(metaclass=abc.ABCMeta):
    def __init__(self,
                 credentials: Any) -> None:
        self._credentials = credentials

    @abc.abstractmethod
    def retrieve_context(self, question: str, **kwargs) -> str:
        raise NotImplementedError()

class APIContextService(ContextService, metaclass=abc.ABCMeta):
    def __init__(self,
                 llm: BaseChatModel,
                 documentation: str,
                 credentials: Any,
                 domains: list[str]) -> None:
        super().__init__(credentials)
        self._llm = llm
        self._documentation = documentation
        self._domains = domains

class RAGContextService(ContextService, metaclass=abc.ABCMeta):
    pass
