# Langchain Core
from langchain_core.messages import BaseMessage

# Langchain
from langchain.prompts import ChatPromptTemplate

class MessageMapper:
    """A class to map messages between different formats.

    This class provides static methods to convert a list of tuple messages to a list of base 
    messages and vice versa.

    Methods:
        to_base_messages_from(tuple_messages: list[tuple[str, str]]) -> list[BaseMessage]: 
            Converts a list of tuple messages to a list of base messages.

        to_tuple_messages_from(base_messages: list[BaseMessage]) -> list[tuple[str, str]]: 
            Converts a list of base messages to a list of tuple messages.
    """
    @staticmethod
    def to_base_messages_from(tuple_messages: list[tuple[str, str]]) -> list[BaseMessage]:
        """Converts a list of tuple messages to a list of base messages.

        This method uses the ChatPromptTemplate to format the tuple messages into a prompt, 
        and then converts this prompt into a list of base messages.

        Args:
            tuple_messages (list[tuple[str, str]]): The list of tuple messages to convert.

        Returns:
            list[BaseMessage]: The converted list of base messages.
        """
        base_messages_prompt = ChatPromptTemplate.from_messages(tuple_messages).format_prompt()
        base_messages = base_messages_prompt.to_messages()
        return base_messages

    @staticmethod
    def to_tuple_messages_from(base_messages: list[BaseMessage]) -> list[tuple[str, str]]:
        """Converts a list of base messages to a list of tuple messages.

        This method iterates over the base messages and for each message, it checks the type of 
        the message and appends a tuple with the type and content of the message to the list of 
        tuple messages.

        Args:
            base_messages (list[BaseMessage]): The list of base messages to convert.

        Returns:
            list[tuple[str, str]]: The converted list of tuple messages.
        """
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
