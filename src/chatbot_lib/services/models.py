import abc
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain.chains import APIChain

class ContextService(metaclass=abc.ABCMeta):
    """An abstract base class for context services.

    This class provides a constructor and an abstract method for retrieving context.

    Methods:
        retrieve_context(question: str, **kwargs) -> str: An abstract method for retrieving context.
    """
    def __init__(self) -> None:
        """Initializes a new instance of the ContextService class."""

    @abc.abstractmethod
    def retrieve_context(self, question: str, **kwargs) -> str:
        """An abstract method for retrieving context.

        Args:
            question (str): The question for which to retrieve the context.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError()

class APIContextService(ContextService):
    """A class for API context services.

    This class extends ContextService and provides a constructor and a method for retrieving 
    context from an API.

    Attributes:
        llm (BaseChatModel): The language learning model of the context service.
        documentation (str): The API documentation.
        credentials (dict[str, Any]): The API credentials.
        headers (list[dict[str, Any]]): The API headers.
        domains (list[str]): The API domains.

    Methods:
        retrieve_context(question: str, **kwargs) -> str: Retrieves context from an API.
    """
    def __init__(self,
                 llm: BaseChatModel,
                 documentation: str,
                 credentials: dict[str, Any],
                 headers: list[dict[str, Any]],
                 domains: list[str]) -> None:
        """Initializes a new instance of the APIContextService class.

        Args:
            llm (BaseChatModel): The language learning model of the context service.
            documentation (str): The API documentation.
            credentials (dict[str, Any]): The API credentials.
            headers (list[dict[str, Any]]): The API headers.
            domains (list[str]): The API domains.
        """
        super().__init__()
        self._llm = llm
        self._documentation = documentation
        self._credentials = credentials
        self._headers = headers
        self._domains = domains

    def retrieve_context(self, question: str, **kwargs) -> str:
        """Retrieves context from an API.

        This method creates an API chain from the language learning model and API documentation, 
        and then invokes this chain with the given question.

        Args:
            question (str): The question for which to retrieve the context.

        Returns:
            str: The context retrieved from the API.
        """
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
    """An abstract base class for RAG context services.

    This class extends ContextService but does not provide any additional methods or attributes.
    """
