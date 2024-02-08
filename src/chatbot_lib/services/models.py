import abc
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain.chains import APIChain

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
                 credentials: dict[str, Any],
                 headers: list[dict[str, Any]],
                 domains: list[str]) -> None:
        super().__init__(credentials)
        self._llm = llm
        self._documentation = documentation
        self._headers = headers
        self._domains = domains

    def retrieve_context(self, question: str, **kwargs) -> str:
        chain = APIChain.from_llm_and_api_docs(
            llm=self._llm,
            api_docs=self._documentation,
            headers=self._headers,
            auth=self._credentials,
            limit_to_domains=self._domains,
            kwargs=kwargs
        )

        return chain.invoke(question)

class RAGContextService(ContextService, metaclass=abc.ABCMeta):
    pass
