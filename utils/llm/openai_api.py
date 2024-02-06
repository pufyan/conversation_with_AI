import datetime
import json
import traceback
from typing import List
from openai import AsyncOpenAI, OpenAI

from utils.llm.common import cut_messages_by_tokens, filter_unanswered_tools, count_tokens, Message
from utils.color_printing import print_g, print_r, print_y
from utils.metrics import MetricWriter
# TODO:
# from utils.llm.settings import API_KEY, MODEL, MAX_TOKENS_ANSWER, TOKENS, TOOLS

import openai
import os
import dotenv
dotenv.load_dotenv(dotenv.find_dotenv())
GPT_KEY = os.getenv('GPT_KEY')
MODEL = os.getenv('MODEL') or 'claude-2'

if GPT_KEY and 'gpt' in MODEL:
    openai.api_key = GPT_KEY

TOKENS = 16385  # TODO use dynamic value
MAX_TOKENS_ANSWER = 5000

TOOLS = []  # TODO

metric_writer = MetricWriter("server_logs")

# @async_crush_on_timeout(60)


def get_response(messages: List[Message], **kwargs):

    # messages = list(sorted(messages, key=lambda x: x.id, reverse=False))
    prompt = kwargs.get('prompt', '') or kwargs.get('system_prompt', '') or ''
    tool_choice = kwargs.get('tool_choice', 'auto')
    model = kwargs.get('model', 'gpt-3.5-turbo-16k')
    debug = kwargs.get('debug', False)

    # An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'
    # find all assistant mesagges with 'tool_calls' and remove them if there are no tool messages with 'tool_call_id' next to them

    # print_r('Sorted objects:', '\n'.join([str(m.dict) for m in messages]))
    if TOOLS:
        tools_tokens = count_tokens(model, json.dumps(TOOLS, indent=2))
    else:
        tools_tokens = 0

    messages = cut_messages_by_tokens(
        messages,
        TOKENS - tools_tokens - MAX_TOKENS_ANSWER - 200,
        system_prompt=prompt,
        model=model,
    )

    messages = filter_unanswered_tools(messages)

    # chat_messages = [m.to_openai_dict() for m in messages]

    # client = AsyncOpenAI(
    #     api_key=API_KEY
    # )

    client = OpenAI(
        api_key=GPT_KEY
    )

    # print_g(
    #     json.dumps(dict(
    #         messages=[{
    #             'role': 'system',
    #             'content': '<SYSTEM_PROMPT>'
    #         }]  + chat_messages,

    #         model="gpt-4-1106-preview",

    #         tool_choice=tool_choice,
    #     ),
    #     indent=4, ensure_ascii=False)
    # )

    chat_messages = []
    if prompt:
        chat_messages.append({
            'role': 'system',
            'content': prompt
        })

    if messages:
        chat_messages += [m.to_openai_dict() for m in messages]

    request = dict(
        messages=chat_messages,
        model=MODEL,
        # tools=TOOLS,
        # tool_choice=tool_choice if TOOLS else None,

        # max_tokens=MAX_TOKENS_ANSWER,
    )

    if debug:
        print_y('\nGPT Request:')
        print(json.dumps(request, indent=4, ensure_ascii=False))
        print_y('-----')

    try:
        start = datetime.datetime.now()
        response = client.chat.completions.create(**request)

        metric_writer.log_metrics(
            model=model,
            method='openai_api.get_response',

            time_elapsed=(datetime.datetime.now() - start).total_seconds(),
            token_count=response.usage.total_tokens,

            input_tokens_llm=response.usage.prompt_tokens,
            output_tokens_llm=response.usage.completion_tokens,

            input_text=json.dumps(request, ensure_ascii=False),
            output_text=response.choices[0].message.content,
        )
    except Exception as e:
        print_r(f'\nERROR\n"{e}"')
        print(json.dumps(request, indent=4, ensure_ascii=False))
        print_y(traceback.format_exc())
        print_r('-----')
        return

    if debug:
        print_y('GPT Response:')
        print(response)
        print_y('-----')

    return response
