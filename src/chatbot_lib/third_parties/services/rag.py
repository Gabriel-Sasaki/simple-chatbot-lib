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
        credentials (Any): The Azure credentials.

    Methods:
        retrieve_context(question: str, **kwargs) -> str: Retrieves context from Azure.
    """
    def __init__(self, credentials: Any) -> None:
        if credentials.get('endpoint') is None:
            raise ValueError('For AzureRAGContextService, the credentials must '\
                'to have the endpoint attribute')
        if credentials.get('azure_key') is None:
            raise ValueError('For AzureRAGContextService, the credentials must '\
                'to have the azure_key attribute')
        if credentials.get('index_name') is None:
            raise ValueError('For AzureRAGContextService, the credentials must '\
                'to have the index_name attribute')
        super().__init__(credentials)

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
        endpoint = self._credentials.get('endpoint')
        azure_key = self._credentials.get('azure_key')
        index_name = self._credentials.get('index_name')
        credential = AzureKeyCredential(azure_key)
        client = SearchClient(endpoint=endpoint,
                            index_name=index_name,
                            credential=credential)
        top = kwargs.get('top', 1)
        search_item_paged = client.search(search_text=question, top=top)
        results = ''
        for item_paged in search_item_paged:
            for key, value in item_paged.items():
                results += f'{key}: {value}\n'
        return results
