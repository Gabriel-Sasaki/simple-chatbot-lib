# Simple Chatbot Lib
This repository contains several modules that can be used to create chatbots capable of interacting with users. Each module serves a specific purpose and provides different functionalities for building chatbot applications.

# Classes
## BaseChatbot
The `BaseChatbot` class provides a base class for creating chatbots. It includes methods for starting a conversation, creating system messages with restrictions, creating prompt messages for the user, and creating introduction messages. This class serves as the foundation for more specialized chatbot classes.

## SimpleChatbot
The `SimpleChatbot` class extends the `BaseChatbot` class and represents a simple chatbot. In addition to the functionalities provided by the `BaseChatbot`, the SimpleChatbot is capable of retrieving context for a given question and creating system messages with the provided contexts. This class is useful for chatbots that require contextual information to provide accurate responses.

## AgentChatbot
The `AgentChatbot` class extends the `BaseChatbot` class and represents an agent chatbot. In addition to the functionalities provided by the `BaseChatbot`, the `AgentChatbot` is capable of choosing tools to answer a user's question, executing a set of tools and returning the results, and creating a results message based on the provided results. This class is useful for chatbots that require access to a set of tools to perform specific tasks.

## ToolResult and ToolSet
The `ToolResult` and `ToolSet` represent the result of a tool execution and a set of tools, respectively. These classes are useful for chatbots that need to execute specific functions or tools to provide answers or perform tasks.

## ContextService and APIContextService
The `ContextService` and `APIContextService` represent classes for retrieving context from various sources. The `ContextService` is an abstract base class that defines a common interface for context services, while the `APIContextService` is a concrete implementation that retrieves context from an API. These classes are useful for chatbots that require external sources of information to provide accurate responses.

## MessageMapper
The `MessageMapper` class provides methods for mapping messages between different formats. It includes methods for converting a list of tuple messages to a list of base messages and vice versa. This class is useful for chatbots that need to work with different message formats.

## AzureAISearchContextService
The `AzureAISearchContextService` class retrieves context from Azure using Azure Cognitive Search. It extends the `ContextService` class and includes methods for retrieving context from Azure. This class is useful for chatbots that need to search for information in an Azure index.

# Usage
To use these classes, simply import the desired module into your chatbot application and create an instance of the corresponding class. You can then use the methods and attributes provided by the class to interact with users and perform specific tasks.

```python
from simple_chatbot_lib.chatbots import SimpleChatbot
from simple_chatbot_lib.mappers import MessageMapper
from simple_chatbot_lib.third_parties.context_services import AzureAISearchContextService
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(openai_api_key='...')

az_ai_search_context = AzureAISearchContextService(
    azure_key='...',
    endpoint='...',
    index_name='...'
)

simple_chatbot = SimpleChatbot(llm=llm,
                               context_services=[az_ai_search_context],
                               restrictions=['Do not answer questions that deviate from the informed context'],
                               personality='Friendly, helpful, and respectful',
                               language='English',
                               base_messages=None,
                               message_mapper=MessageMapper())

response = simple_chatbot('What are your business hours?')
print(response)
```

# Requirements
* Python >= 3.11
* LangChain >= 0.1.0 < 0.2.0

# License

This project is licensed under the terms of the GNU General Public License v3.0. See the [LICENSE](LICENSE.md) file for details.