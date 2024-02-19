# Chatbot Context Service

This project consists of a set of classes that represent a chatbot and various context services. The chatbot uses a language learning model to interact with users and retrieve context for their questions. The context services provide different methods for retrieving context, such as through an API or from Azure.

## Classes

### Chatbot

The `Chatbot` class represents a chatbot. It has methods to initiate a chat with an AI, retrieve the context for a given question, create system and human prompt messages, and retrieve the base messages and tuple messages.

### ContextService

The `ContextService` class is an abstract base class for context services. It provides a constructor and an abstract method for retrieving context.

### APIContextService

The `APIContextService` class extends `ContextService` and provides a constructor and a method for retrieving context from an API.

### AzureRAGContextService

The `AzureRAGContextService` class extends `RAGContextService` and provides a constructor and a method for retrieving context from Azure Cognitive Search.

## Usage

To use these classes, you need to create instances of them and call their methods. For example, to create a chatbot and initiate a chat, you can do:

```python
from chatbot_lib.chatbots.models import Chatbot
from chatbot_lib.mappers.messages import MessageMapper
from chatbot_lib.services.models import APIContextService
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(openai_api_key='...')

azure_rag_context = AzureRAGContextService(
    azure_key='...',
    endpoint='...',
    index_name='...'
)

message_mapper = MessageMapper()

chatbot = Chatbot(llm=llm,
                  context_services=[azure_rag_context],
                  restrictions=['Do not answer questions that deviate from the informed context'],
                  personality='Friendly, helpful, and respectful',
                  language='English',
                  base_messages=None,
                  message_mapper=message_mapper)

response = chatbot('What are your business hours?')
print(response)
```

## Requirements

This project requires Python 3.11 or later. Several of the features are built on top of the LangChain 0.1.0 library.

## License

This project is licensed under the terms of the GNU General Public License v3.0. See the [LICENSE](LICENSE.md) file for details.