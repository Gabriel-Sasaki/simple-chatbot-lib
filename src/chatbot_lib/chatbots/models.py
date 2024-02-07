# Python
from typing import Optional

# Langchain Core
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel

# Langchain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Chatbot Lib
from services.models import ContextService
from mappers.messages import MessageMapper

class Chatbot:
    def __init__(self,
                 llm: BaseChatModel,
                 context_services: list[ContextService],
                 restrictions: list[str],
                 personality: str,
                 base_messages: Optional[list[BaseMessage]],
                 message_mapper: MessageMapper) -> None:
        if len(context_services) == 0:
            raise ValueError('The context_services argument must to have one or more values')
        if len(restrictions) == 0:
            raise ValueError('The restrictions argument must to have one or more values')

        self.__llm = llm
        self.__context_services = context_services
        self.__restrictions = restrictions
        self.__personality = personality
        self.__message_mapper = message_mapper
        
        if base_messages is None:
            self.__base_messages = [self.__create_introduction()]
        else:
            self.__base_messages = base_messages

    @property
    def tuple_messages(self) -> list[tuple[str, str]]:
        mapper = self.__message_mapper
        base_messages = self.__base_messages
        return mapper.to_tuple_messages_from(base_messages)
    
    @property
    def base_messages(self) -> list[BaseMessage]:
        return self.__base_messages

    def chat(self, question: str) -> str:
        contexts = self.__retrieve_context(question)
        system_message = self.__create_system_prompt(contexts)
        human_message = self.__create_human_prompt(question)
        self.__base_messages.append(system_message)
        self.__base_messages.append(human_message)
        ai_message = self.__llm.invoke(input=self.__base_messages)
        self.__base_messages.append(ai_message)
        return str(ai_message.content)

    def __create_system_prompt(self, contexts) -> str:
        template = """Responda a dúvida do usuário utilizando o contexto entre as
        tags <context></context> e obedecendo as restrições entre as
        tags <restrictions></restrinctions>
        
        <context>
        {context}
        </context>
        
        <restrictions>
        {restrictions}
        </restrictions>
        """
        final_context = ''
        for index, context in enumerate(contexts):
            final_context += f'{index+1} - {context}\n'
        final_restrictions = ''
        for index, restriction in enumerate(self.__restrictions):
            final_restrictions += f'{index+1} - {restriction}\n'
        system_prompt_template = SystemMessagePromptTemplate.from_template(template)
        message = system_prompt_template.format(context=final_context,
                                                restrictions=final_restrictions)
        return message

    def __create_human_prompt(self, question: str) -> BaseMessage:
        human_prompt_template = HumanMessagePromptTemplate.from_template(question)
        return human_prompt_template.format()

    def __create_introduction(self) -> BaseMessage:
        template = 'Você é um chatbot {personality}'
        system_prompt_template = SystemMessagePromptTemplate.from_template(template)
        message = system_prompt_template.format(personality=self.__personality)
        return message

    def __retrieve_context(self, question: str) -> list[str]:
        contexts = []
        for context_service in self.__context_services:
            context = context_service.retrieve_context(question)
            contexts.append(context)
        return contexts

    def __call__(self, question: str) -> str:
        return self.chat(question)