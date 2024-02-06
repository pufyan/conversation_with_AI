
import json
import datetime
import sys
import os
import re
import random
import tiktoken
from typing import Dict
from utils.color_printing import *

__all__ = [
    'Message',
    'filter_unanswered_tools',
    'count_tokens',
    'cut_messages_by_tokens',
]

MODEL = os.getenv('MODEL') or 'claude-2'

# SAFE_SPACE_TOKEN_COEF = 0.95
# LLM_TOKENS_LIMIT = {
#     'gpt-4-0125-preview': 80000,
#     'gpt-4-1106-preview': 80000,
#     'gpt-3.5-turbo': 4096,
#     'gpt-3.5-turbo-16k': 16384,
#     'gpt-4': 8192,
#     'claude-instant-1': 70000,
#     'claude-2': 70000,
# }.get(MODEL, 4096) * SAFE_SPACE_TOKEN_COEF  # type: ignore


class Message:
    def __init__(
        self,
        user_id=None, content=None, role=None,
        tool_calls=None, function_call=None,
        data=None, date_time=None, id=None
    ):
        self.user_id = user_id
        self.content = content
        self.role = role
        # self.tool_calls = tool_calls if tool_calls is not None else []
        self.tool_calls = tool_calls
        self.function_call = function_call
        self.data = data if data is not None else {}
        self.date_time = date_time if date_time else datetime.datetime.utcnow()
        self.id = id or random.randint(0, sys.maxsize)

    @property
    def dict(self):
        return self.to_dict()

    @property
    def tool_call_ids(self):
        if self.tool_calls:
            return [tool_call['id'] for tool_call in self.tool_calls]
        elif 'tool_call_id' in self.data:
            return [self.data['tool_call_id']]
        return []

    @staticmethod
    def from_openai_chat_message(msg, user_id: int):
        return Message(
            user_id=user_id,
            content=msg.get('content'),
            role=msg.get('role'),
            tool_calls=msg.get('tool_calls'),
            function_call=msg.get('function_call'),
            data=msg,
            date_time=datetime.datetime.utcnow()
        )

    # @staticmethod
    # def from_dict(d: dict):
    #     return Message(**d)

    def to_dict(self):
        return {
            'user_id': self.user_id,
            'content': self.content,
            'role': self.role,
            'tool_calls': self.tool_calls,
            'function_call': self.function_call,
            'data': self.data,
            'date_time': self.date_time.isoformat()
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    def __str__(self):
        return self.to_json()

    @staticmethod
    def from_dict(d: Dict):
        msg_obj = Message()
        msg_obj.data = {}

        for key in d:
            setattr(msg_obj, key, d[key])
            # if key in Message.__table__.columns.keys():
            #     setattr(msg_obj, key, d[key])
            # else:
            #     msg_obj.data[key] = d[key]

        return msg_obj

    def to_openai_dict(self):
        result = self.dict

        if self.data:
            result.update(self.data)

        result.pop('id', None)
        result.pop('user_id', None)
        result.pop('date_time', None)
        result.pop('data', None)

        for key in list(result.keys()):
            if result[key] is None:
                result.pop(key)

        # convert tool_calls into list of namedtuple objects with recursive conversion:

        # if result.get('tool_calls'):
        #     old = result['tool_calls'].copy()
        #     result['tool_calls'] = []
        #     for tc in old:
        #         tcObject = namedtuple('ToolCall', tc.keys())(*tc.values())
        #         tcObject = tcObject._replace(
        #             function_call = namedtuple('FunctionCall', tcObject.function_call.keys())(*tcObject.function_call.values())
        #         )
        #         tcObject = tcObject._replace(
        #             function_call = tcObject.function_call._replace(
        #                 data = namedtuple('Data', tcObject.function_call.data.keys())(*tcObject.function_call.data.values())
        #             )
        #         )
        #         tcObject = tcObject._replace(
        #             function_call = tcObject.function_call._replace(
        #                 tool_calls = [namedtuple('ToolCall', tcObject.function_call.tool_calls[0].keys())(*tcObject.function_call.tool_calls[0].values())]
        #             )
        #         )
        #         result['tool_calls'].append(tcObject)

        # print(result)
        # print('\n\n')
        return result


def filter_unanswered_tools(messages):
    if len(messages) in [0, 1]:
        return messages

    messages = messages.copy()
    tool_calls = dict()

    for i in range(len(messages)):

        if messages[i].role == 'tool':
            # print('call', messages[i].tool_call_ids)
            for t in messages[i].tool_call_ids:
                tool_calls[f"{t}_call"] = i

        if messages[i].role == 'assistant' and messages[i].tool_calls:
            # print('resp', messages[i].tool_call_ids)
            for t in messages[i].tool_call_ids:
                tool_calls[f"{t}_response"] = i

    to_remove = set()
    unresponsed_tool_calls = set()
    uncalled_tool_responses = set()

    for t in tool_calls:
        if '_call' in t and t.replace('_call', '_response') not in tool_calls:
            unresponsed_tool_calls.add(tool_calls[t])
            to_remove.add(tool_calls[t])

        if '_response' in t and t.replace('_response', '_call') not in tool_calls:
            uncalled_tool_responses.add(tool_calls[t])
            to_remove.add(tool_calls[t])

    # print('unresponsed_tool_calls:', unresponsed_tool_calls)
    # print('uncalled_tool_responses:', uncalled_tool_responses)
    # print('to_remove:', to_remove)

    if to_remove:
        messages = [m for i, m in enumerate(messages) if i not in to_remove]
        return filter_unanswered_tools(messages)

    return messages


def count_tokens(model, text: str) -> int:
    if 'claude' in model:
        # TODO
        # return client.count_tokens(text)
        raise NotImplementedError('TODO CLaude')

    if 'gpt' in model:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))

    raise ValueError(f"Unknown model: {model}")


def cut_messages_by_tokens(messages, request_tokens_limit, system_prompt=None, model=None):
    model = model or MODEL
    system_prompt = system_prompt or ''
    if len(messages) == 1:
        return messages

    if len(messages) == 0:
        return []

    if len(messages) == 2:
        return messages

    print_y(f'Cutting {len(messages)} messages by tokens: {request_tokens_limit}')

    _result = messages.copy()

    tokens = None  # supress warning

    while len(_result) > 1:
        tokens = count_tokens(model, json.dumps(
            [{
                'role': 'system',
                'content': system_prompt
            }] + [m.to_openai_dict() for m in _result],
            indent=4,
        ))
        if tokens < request_tokens_limit:
            break

        if tokens > request_tokens_limit * 1.2:
            to_remove = int(max(1, len(_result) * 0.2))

        else:
            to_remove = 1

        print_y('Tokens count:', tokens, f'too much ({request_tokens_limit}), removing {to_remove} msgs')
        # if len(_result):
        #     print(f"1st message: \"{len(_result) and _result[0].to_openai_dict()}\"")

        _result = _result[to_remove:]

        if len(_result) == 1:
            print('Tokens count:', tokens, f'too much ({request_tokens_limit}) for one message, len(__messages) == ', len(_result))
            # TODO check if it's a system message
            # TODO check token count
            # TODO throw error or summarize
            raise Exception('Tokens count: {} too much for one message, len(__messages) == {}'.format(tokens, len(_result)))

    print_g('Tokens count:', tokens, f'ok ({request_tokens_limit})')
    return _result
