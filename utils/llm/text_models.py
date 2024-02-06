# encoding: utf-8

import json
import tiktoken
import backoff
import openai
import requests
import dotenv
import os
import sys
import asyncio


for i in range(2):
    try:
        from utils.color_printing import *
        from utils.metrics import MetricWriter
        from utils.misc import async_crush_on_timeout
        from utils.llm.openai_api import Message, get_response as get_gpt_response
        from utils.llm.claude_api import ClaudeApi
        from utils.llm.common import count_tokens

    except ImportError:
        sys.path.append(os.getcwd())
    else:
        break


__all__ = [
    'get_llm_response',
    'transcribe_array_of_uint8',
    'transcribe',
]


# TODO remove
dotenv.load_dotenv(dotenv.find_dotenv())
MODEL = os.getenv('MODEL') or 'claude-2'
PROMPT = ''
TEMPERATURE = 0.8
DEBUG = False


def get_chat(_messages, before=None, after=None):
    _messages = _messages.copy()
    if before:
        if isinstance(before, list):
            before_messages = before
        else:
            before_messages = [{
                'role': 'system',
                'content': str(before),
            }]
        _messages = before_messages + _messages

    if after:
        _messages += [{
            'role': 'system',
            'content': str(after),
        }]
    return _messages


def print_error_decorator(func):
    import sys

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print('Error in {}: {}'.format(func.__name__, e))
            print('args: ', args)
            print('kwargs: ', kwargs)

            raise e

        except KeyboardInterrupt:
            print('KeyboardInterrupt in {}'.format(func.__name__))
            sys.exit(1)

        # except openai.error.InvalidRequestError as e:
        #     print('openai.error.InvalidRequestError in {}: {}'.format(func.__name__, e))
        #     print('args: ', args)
        #     print('kwargs: ', kwargs)

        #     raise e
    return wrapper


# @backoff.on_exception(
#     backoff.expo,
#     Exception,
#     max_tries=5,
# )
def get_llm_response(
    messages,
    debug=DEBUG,

    # prompting:
    prompt=None,
    context_info=None,

    # api:
    max_tokens=None,
    functions=None,
    function_call=None,
    temperature=TEMPERATURE,
    model=MODEL,
    api_key=None,
    **kwargs,
) -> str:

    messages = [
        Message(**m) if isinstance(m, dict) else m for m in messages
    ]

    # print_pink(f'Selected Model: {model}, debug is {debug}')

    if model in ['claude-instant-1', 'claude-2']:
        resp = ClaudeApi.get_response(
            messages,
            debug=debug,
            prompt=prompt,
            context_info=context_info,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
        )
        return resp

    elif 'gpt' in model:
        resp = get_gpt_response(
            messages,
            debug=debug,
            prompt=prompt,
            context_info=context_info,
            max_tokens=max_tokens,
            functions=functions,
            function_call=function_call,
            temperature=temperature,
            model=model,
            # api_key=api_key or GPT_KEY,
            **kwargs,
        )
        # TODO refactor and add features

        return resp.choices[0].message.content

    raise NotImplementedError('Model {} not implemented'.format(model))
    _messages = messages
    prompt = prompt or PROMPT
    while 1:
        tokens = count_tokens(model, str(get_chat(_messages, after=prompt, before=context_info)))
        if tokens > LLM_TOKENS_LIMIT and len(_messages) > 1:
            if tokens > LLM_TOKENS_LIMIT * 1.5:
                to_remove = int(max(1, len(_messages) * 0.3))
            else:
                to_remove = 1
            print('Tokens count:', tokens, 'too much, removing {} message'.format(to_remove))
            _messages = _messages[to_remove:]
        else:
            break
    # print('messages after')
    # print(messages)

    request = dict(
        model=model,
        messages=[],
        temperature=float(temperature),
        max_tokens=max_tokens,
    )

    if function_call:
        request['function_call'] = function_call

    if functions:
        request['functions'] = functions

    for msg in get_chat(_messages, after=prompt, before=context_info):
        message = {
            'role': msg['role'],
            'content': str(msg['content'] or ' '),
        }

        request['messages'].append(message)

    if debug:
        print('OPEANAI REQUEST:')
        print(json.dumps(request, indent=2, ensure_ascii=False))
        # await log('SYSTEM', f"sending request to openai: {_messages}")

    resp = openai.ChatCompletion.create(
        **request
    )

    if debug:
        print('OPEANAI RESPONSE:')
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        # await log('SYSTEM', f"response from openai: {resp.choices[0]['message']['content']}")

    if functions:
        return resp
    return resp.choices[0]['message']['content']


def transcribe_array_of_uint8(arr):
    with open('tmp.mp3', 'wb') as f:
        f.write(arr)
    return transcribe('tmp.mp3')


def transcribe(mp3_file):
    with open(mp3_file, "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)

    print(json.dumps(transcript, indent=2, ensure_ascii=False))
    text = transcript.get('text', '')

    return text


if __name__ == '__main__':
    ping_response = get_llm_response([{
        'role': 'user',
        'content': 'this is a ping, answer pong'
    }])
