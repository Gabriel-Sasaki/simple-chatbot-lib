# Python
from typing import Any

# Azure
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

# Chatbot lib
from chatbot_lib.services.models import RAGContextService

class AzureRAGContextService(RAGContextService):
    """A class for Azure RAG context services.

    This class extends RAGContextService and provides a constructor and a method for retrieving 
    context from Azure.

    Attributes:
        _endpoint (str): The Azure endpoint.
        _azure_key (str): The Azure key.
        _index_name (str): The Azure index name.

    Methods:
        retrieve_context(question: str, **kwargs) -> str: Retrieves context from Azure.
    """
    def __init__(self, endpoint: str, azure_key: str, index_name: str) -> None:
        """Initializes the AzureRAGContextService with the given endpoint, 
        azure_key, and index_name.

        Args:
            endpoint (str): The Azure endpoint.
            azure_key (str): The Azure key.
            index_name (str): The Azure index name.
        """
        super().__init__()
        self._endpoint = endpoint
        self._azure_key = azure_key
        self._index_name = index_name

    def retrieve_context(self, question: str, **kwargs) -> str:
        """Retrieves context from Azure.

        This method uses the Azure credentials to create a SearchClient, and then uses this client 
        to search for the given question. The search results are formatted into a string and 
        returned.

        Args:
            question (str): The question for which to retrieve the context.

        Returns:
            str: The context retrieved from Azure.
        """
        credential = AzureKeyCredential(self._azure_key)
        client = SearchClient(endpoint=self._endpoint,
                            index_name=self._azure_key,
                            credential=credential)
        top = kwargs.get('top', 1)
        search_item_paged = client.search(search_text=question, top=top)
        results = ''
        for item_paged in search_item_paged:
            for key, value in item_paged.items():
                results += f'{key}: {value}\n'
        return results
