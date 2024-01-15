# Langchain Core
from langchain_core.messages import BaseMessage

# Langchain
from langchain.prompts import ChatPromptTemplate

class MessageMapper:
    @staticmethod
    def to_base_messages_from(tuple_messages: list[tuple[str, str]]) -> list[BaseMessage]:
        base_messages = ChatPromptTemplate.from_messages(tuple_messages).format_prompt().to_messages()
        return base_messages

    @staticmethod
    def to_tuple_messages_from(base_messages: list[BaseMessage]) -> list[tuple[str, str]]:
        tuple_messages = []
        for message in base_messages:
            match message.type:
                case 'human':
                    message_type = 'user'
                case 'ai':
                    message_type = 'assistant'
                case _:
                    message_type = message.type
            tuple_messages.append((message_type, message.content))
        return tuple_messages
