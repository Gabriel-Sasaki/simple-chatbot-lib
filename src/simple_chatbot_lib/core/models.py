"""This module contains the classes ToolResult and ToolSet, which
are used to represent the result of a tool execution and a set of tools,
respectively.

The ToolResult class represents the result of a tool execution. It
has attributes tool_name and tool_arguments, which store the name
of the tool and a dictionary containing the tool arguments, respectively.

The ToolSet class represents a set of tools. It has an attribute
tool_results, which is a list of ToolResult objects containing
the tool name and tool arguments.
"""
# Langchain
from langchain.pydantic_v1 import BaseModel, Field

class ToolResult(BaseModel):
    """Represents the result of a tool execution.

    Attributes:
        tool_name (str): The name of the tool.
        tool_arguments (dict): A dictionary containing the tool arguments,
            where the keys are the argument names and the values are the
            argument values.
    """
    tool_name: str = Field(description='The tool name')
    tool_arguments: dict = Field(description='''A dict
                                    containing the key as
                                    the argument name and
                                    value as the argument
                                    value''')

class ToolSet(BaseModel):
    """Represents a set of tools.

    Attributes:
        tool_results (list[ToolResult]): A list of ToolResult objects
        containing tool_name and tool_arguments.
    """
    tool_results: list[ToolResult] = Field(description=
                                           '''A list of ToolResult
                                           containing tool_name and
                                           tool_arguments''')
