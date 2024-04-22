"""This module contains the classes BaseChatbot, SimpleChatbot, and 
AgentChatbot, which are used to create chatbots capable of interacting with users.

The BaseChatbot class is an abstract class that represents a basic
chatbot. It has methods for starting a conversation, creating system messages with
restrictions, creating prompt messages for the user, and creating introduction messages.

The SimpleChatbot class is a class that extends the BaseChatbot class and
represents a simple chatbot. In addition to the functionalities of the BaseChatbot
class, it is also capable of retrieving context for a given question and creating
system messages with the provided contexts.

The AgentChatbot class is a class that extends the BaseChatbot class and
represents an agent chatbot. In addition to the functionalities of the BaseChatbot
class, it is also capable of choosing tools to answer a user's question, executing
a set of tools and returning the results, and creating a results message based on the
provided results.
"""
# Python
import abc
import json
from typing import Optional

# Langchain Core
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage

# Langchain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser

# Chatbot Lib
from simple_chatbot_lib.context_services import ContextService
from simple_chatbot_lib.mappers import MessageMapper
from simple_chatbot_lib.core.models import ToolSet

class BaseChatbot(metaclass=abc.ABCMeta):
    """A BaseChatbot class that uses the abc.ABCMeta metaclass. This class
    represents a base chatbot that can chat with users, 
    create system message prompts with restrictions, create human prompts,
    and create introduction messages.

    Attributes:
        llm (BaseChatModel): The language model used by the chatbot.
        restrictions (list[str]): A list of restrictions for the chatbot.
        personality (str): The personality of the chatbot.
        language (str): The language used by the chatbot.
        message_mapper (MessageMapper): The message mapper used by the chatbot.
        keep_messages (bool, optional): Whether to keep the chat messages.
        Defaults to True.
        base_messages (Optional[list[BaseMessage]], optional): A list of
        base messages. Defaults to None.

    Methods:
        chat(question: str) -> str: 
            Chat with the chatbot by providing a question. Returns the chatbot's
            response to the question.
        _create_restrictions_message() -> BaseMessage: 
            Creates a system message prompt with the restrictions. Returns the
            system message prompt with the restrictions.
        _create_human_prompt(question: str) -> BaseMessage: 
            Creates a human prompt for the chatbot. Returns the human prompt message.
        _create_introduction() -> BaseMessage: 
            Creates an introduction message for the chatbot. Returns the introduction
            message.
    """
    def __init__(self,
                 llm: BaseChatModel,
                 restrictions: list[str],
                 personality: str,
                 language: str,
                 message_mapper: MessageMapper,
                 keep_messages: bool = True,
                 base_messages: Optional[list[BaseMessage]] = None) -> None:
        """Initializes a Chatbot object.

        Args:
            llm (BaseChatModel): The language model used by the chatbot.
            restrictions (list[str]): A list of restrictions for the chatbot.
            personality (str): The personality of the chatbot.
            language (str): The language used by the chatbot.
            message_mapper (MessageMapper): The message mapper used by the chatbot.
            keep_messages (bool, optional): Whether to keep the chat messages.
            Defaults to True.
            base_messages (Optional[list[BaseMessage]], optional): A list of base
            messages. Defaults to None.

        Returns:
            None
        """
        self._llm = llm
        self._restrictions = restrictions
        self._personality = personality
        self._language = language
        self._message_mapper = message_mapper
        self._keep_messages = keep_messages

        if base_messages is None or len(base_messages) < 1:
            self._base_messages = [self._create_introduction()]
        else:
            self._base_messages = base_messages

    @property
    def tuple_messages(self) -> list[tuple[str, str]]:
        """Returns a list of tuples representing the messages in the chatbot.

        Returns:
            A list of tuples, where each tuple contains two strings representing 
            a message in the chatbot.
        """
        mapper = self._message_mapper
        base_messages = self._base_messages
        return mapper.to_tuple_messages_from(base_messages)

    @property
    def base_messages(self) -> list[BaseMessage]:
        """Returns the list of base messages for the chatbot.

        Returns:
            A list of BaseMessage objects representing the base messages for the chatbot.
        """
        return self._base_messages

    def chat(self, question: str) -> str:
        """Chat with the chatbot.

        Args:
            question (str): The question to ask the chatbot.

        Returns:
            str: The response from the chatbot.
        """
        restrictions_system_message = self._create_restrictions_message()
        human_message = self._create_human_prompt(question)
        self._base_messages.append(restrictions_system_message)
        self._base_messages.append(human_message)
        ai_message = self._llm.invoke(input=self._base_messages)
        if self._keep_messages:
            self._base_messages.append(ai_message)
        else:
            self._base_messages.clear()
        return str(ai_message.content)

    def _create_restrictions_message(self) -> BaseMessage:
        """Creates a system message prompt with the restrictions.

        Returns:
            BaseMessage: The system message prompt with the restrictions.
        """
        template = """
        ---
        {restrictions}
        ---

        Obey the restrictions present in the text above.
        """
        restrictions_str = ''
        for index, restriction in enumerate(self._restrictions):
            restrictions_str += f'{index+1} - {restriction}\n'
        system_prompt_template = SystemMessagePromptTemplate.from_template(template)
        return system_prompt_template.format(restrictions=restrictions_str)

    def _create_human_prompt(self, question: str) -> BaseMessage:
        """Creates a human prompt for the chatbot.

        Args:
            question (str): The question to be used in the human prompt.

        Returns:
            BaseMessage: The human prompt message.

        """
        human_prompt_template = HumanMessagePromptTemplate.from_template(question)
        return human_prompt_template.format()

    def _create_introduction(self) -> BaseMessage:
        """Creates an introduction message for the chatbot.

        Returns:
            BaseMessage: The introduction message.

        """
        template = """You are a chatbot with this personality: {personality}.
        Provide as much detail as possible. Answer in {language}.
        """
        system_prompt_template = SystemMessagePromptTemplate.from_template(template)
        message = system_prompt_template.format(personality=self._personality,
                                                language=self._language)
        return message

    def __call__(self, question: str) -> str:
        return self.chat(question)


class SimpleChatbot(BaseChatbot):
    """A SimpleChatbot class that extends the BaseChatbot class. This class
    represents a simple chatbot that can chat with users, 
    retrieve context for a given question, and create system message
    prompts with the given contexts.

    Attributes:
        llm (BaseChatModel): The language model used by the chatbot.
        restrictions (list[str]): A list of restrictions for the chatbot.
        personality (str): The personality of the chatbot.
        language (str): The language used by the chatbot.
        message_mapper (MessageMapper): The message mapper used by the chatbot.
        keep_messages (bool, optional): Whether to keep the chatbot's messages.
        Defaults to True.
        context_services (list[ContextService], optional): A list of context
        services used by the chatbot. Defaults to None.
        base_messages (list[BaseMessage], optional): A list of base messages
        used by the chatbot. Defaults to None.

    Methods:
        chat(question: str) -> str: 
            Chat with the chatbot by providing a question. Returns the chatbot's
            response to the question.
        _create_contexts_message(contexts: list[str]) -> BaseMessage: 
            Creates a system message prompt with the given contexts. Returns the
            system message prompt with the contexts.
        _retrieve_context(question: str) -> list[str]: 
            Retrieves the context for the given question. Returns a list of
            contexts retrieved from the context services.
    """
    def __init__(self,
                 llm: BaseChatModel,
                 restrictions: list[str],
                 personality: str,
                 language: str,
                 message_mapper: MessageMapper,
                 keep_messages: bool = True,
                 context_services: Optional[list[ContextService]] = None,
                 base_messages: Optional[list[BaseMessage]] = None) -> None:
        """Initializes a SimpleChatbot object.

        Args:
            llm (BaseChatModel): The language model used by the chatbot.
            restrictions (list[str]): A list of restrictions for the chatbot.
            personality (str): The personality of the chatbot.
            language (str): The language used by the chatbot.
            message_mapper (MessageMapper): The message mapper used by the chatbot.
            keep_messages (bool, optional): Whether to keep the chatbot's messages.
            Defaults to True.
            context_services (list[ContextService], optional): A list of context
            services used by the chatbot. Defaults to None.
            base_messages (list[BaseMessage], optional): A list of base messages
            used by the chatbot. Defaults to None.
        """
        super(SimpleChatbot, self).__init__(llm, restrictions, personality, language, 
                                            message_mapper, keep_messages, base_messages)
        self._context_services = context_services

    def chat(self, question: str) -> str:
        """Chat with the chatbot by providing a question.

        Args:
            question (str): The question to ask the chatbot.

        Returns:
            str: The chatbot's response to the question.
        """
        has_context = self._context_services is not None and len(self._context_services) > 0
        if has_context:
            contexts = self._retrieve_context(question)
            contexts_system_message = self._create_contexts_message(contexts)
            self._base_messages.append(contexts_system_message)
        restrictions_system_message = self._create_restrictions_message()
        human_message = self._create_human_prompt(question)
        self._base_messages.append(restrictions_system_message)
        self._base_messages.append(human_message)
        ai_message = self._llm.invoke(input=self._base_messages)
        if self._keep_messages:
            self._base_messages.append(ai_message)
        else:
            self._base_messages.clear()
        return str(ai_message.content)

    def _create_contexts_message(self, contexts: list[str]) -> BaseMessage:
        """Creates a system message prompt with the given contexts.

        Args:
            contexts (list[str]): A list of contexts to include in the message.

        Returns:
            BaseMessage: The system message prompt with the contexts.

        """
        template = """
        ---
        {contexts}
        ---

        Use the information present in the text to answer all questions.
        """
        contexts_str = ''
        for index, context in enumerate(contexts):
            contexts_str += f'{index+1} - {context}\n'
        system_prompt = SystemMessagePromptTemplate.from_template(template)
        return system_prompt.format(contexts=contexts_str)

    def _retrieve_context(self, question: str) -> list[str]:
        """Retrieves the context for the given question.

        Args:
            question (str): The question for which the context needs to be retrieved.

        Returns:
            list[str]: A list of contexts retrieved from the context services.

        """
        contexts = []
        for context_service in self._context_services:
            context = context_service.retrieve_context(question)
            contexts.append(context)
        return contexts


class AgentChatbot(BaseChatbot):
    """A AgentChatbot class that extends the BaseChatbot class. This class
    represents an agent chatbot that can chat with users, 
    choose tools to answer a user's question, execute a set of tools and
    return the results, and create a results message based 
    on the provided results.

    Attributes:
        llm (BaseChatModel): The language model used by the chatbot.
        restrictions (list[str]): A list of restrictions for the chatbot.
        personality (str): The personality of the chatbot.
        language (str): The language used by the chatbot.
        message_mapper (MessageMapper): The message mapper used by the chatbot.
        tools (list[StructuredTool]): The list of tools available to the chatbot.
        keep_messages (bool, optional): Whether to keep the chatbot's messages.
        Defaults to True.
        base_messages (list[BaseMessage], optional): A list of base messages used
        by the chatbot. Defaults to None.

    Methods:
        chat(question: str) -> str: 
            Chat with the chatbot by providing a question. Returns the chatbot's
            response to the question.
        _choose_tools(question: str) -> list[StructuredTool]: 
            Choose tools to answer the user's question. Returns a list of selected
            tools.
        _execute_tools(toolset: ToolSet) -> list: 
            Executes a set of tools and returns the results. Returns a list of
            dictionaries containing the executed function 
            name and its result.
        _create_results_message(results: list[dict]) -> BaseMessage: 
            Creates a results message based on the provided results. Returns the
            results message.
    """
    def __init__(self,
                 llm: BaseChatModel,
                 restrictions: list[str],
                 personality: str,
                 language: str,
                 message_mapper: MessageMapper,
                 tools: list[StructuredTool],
                 keep_messages: bool = True,
                 base_messages: Optional[list[BaseMessage]] = None) -> None:
        """Initializes an AgentChatbot object.

        Args:
            llm (BaseChatModel): The language model used by the chatbot.
            restrictions (list[str]): The list of restrictions for the chatbot.
            personality (str): The personality of the chatbot.
            language (str): The language used by the chatbot.
            message_mapper (MessageMapper): The message mapper used by the chatbot.
            tools (list[StructuredTool]): The list of tools available to the chatbot.
            keep_messages (bool, optional): Whether to keep the chatbot's messages.
            Defaults to True.
            base_messages (Optional[list[BaseMessage]], optional): The base messages
            for the chatbot. Defaults to None.
        """
        super(AgentChatbot, self).__init__(llm, restrictions, personality, language,
                                           message_mapper, keep_messages, base_messages)
        self.tools = {}
        for tool in tools:
            if not tool.__doc__:
                raise ValueError(f'Tool {tool.name} must have a docstring')
            self.tools[tool.name] = tool

    def chat(self, question: str) -> str:
        """Chat with the chatbot by asking a question.

        Args:
            question (str): The question to ask the chatbot.

        Returns:
            str: The response from the chatbot.

        """
        if self.tools:
            toolset = self._choose_tools(question)
            results = self._execute_tools(toolset)
            results_message = self._create_results_message(results)
            self.base_messages.append(results_message)
        restrictions_system_message = self._create_restrictions_message()
        human_message = self._create_human_prompt(question)
        self._base_messages.append(restrictions_system_message)
        self._base_messages.append(human_message)
        ai_message = self._llm.invoke(input=self._base_messages)
        if self._keep_messages:
            self._base_messages.append(ai_message)
        else:
            self._base_messages.clear()
        return str(ai_message.content)

    def _choose_tools(self, question: str) -> list[StructuredTool]:
        """Choose tools to answer the user's question.

        Args:
            question (str): The user's question.

        Returns:
            list[StructuredTool]: A list of selected tools.

        """
        template = """
        ---
        FUNCTIONS
        {tools}
        ---

        Above are Python functions. Analyze the description of each one
        and provide the name of one or more functions that can answer the
        user's question.

        ---
        USER QUESTION
        {question}
        ---

        Follow these instructions to answer.

        ---
        INSTRUCTIONS
        {format_instructions}
        ---
        """
        system_prompt = SystemMessagePromptTemplate.from_template(template)
        str_tools = ''
        for tool in self.tools.values():
            function_schema = {
                'name': tool.name,
                'description': tool.description,
                'arguments': tool.args
            }
            str_tools += json.dumps(function_schema) + '\n\n'
        output_parser = PydanticOutputParser(pydantic_object=ToolSet)
        format_instructions = output_parser.get_format_instructions()
        messages = system_prompt.format_messages(tools=str_tools,
                                                 question=question,
                                                 format_instructions=format_instructions)
        human_message = HumanMessage(content=question)
        messages.append(human_message)
        chain = self._llm | output_parser
        return chain.invoke(messages)

    def _execute_tools(self, toolset: ToolSet):
        """Executes a set of tools and returns the results.

        Args:
            toolset (ToolSet): The set of tools to execute.

        Returns:
            list: A list of dictionaries containing the executed function name and its result.

        """
        results = []
        for tool_result in toolset.tool_results:
            tool = self.tools[tool_result.tool_name]
            result = tool.func(**tool_result.tool_arguments)
            results.append({'function_executed': tool_result.tool_name, 'result': result})
        return results

    def _create_results_message(self, results: list[dict]) -> BaseMessage:
        """Creates a results message based on the provided results.

        Args:
            results (list[dict]): A list of dictionaries representing the results of
            executed functions.

        Returns:
            BaseMessage: The results message.
        """
        template = """
        ---
        {results}
        ---

        Above are the functions executed to answer the
        user's question and their results. Use this
        information to respond to the user.
        """
        results_str = ''
        for result in results:
            results_str += f'{json.dumps(result)}\n'
        system_prompt = SystemMessagePromptTemplate.from_template(template)
        return system_prompt.format(results=results_str)
