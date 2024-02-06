import json
import requests
import os

import dotenv


DEBUG = False
TEMPERATURE = 0.8
MODEL = 'claude-instant-1'
PROMPT = ''
CLAUDE_KEY = os.getenv('CLAUDE_KEY')


# def claude_role(role): return 'Human' if role == 'user' else 'Assistant'
def claude_role(role): return role


class ClaudeApi:

    @staticmethod
    # @backoff.on_exception(
    #     backoff.expo,
    #     Exception,
    #     max_tries=5,
    # )
    def get_response(
        messages,
            debug=DEBUG,
            prompt=None,
            context_info=None,
            max_tokens=None,
            temperature=TEMPERATURE,
            model=MODEL,
            api_key=CLAUDE_KEY,
    ):
        prompt = prompt or PROMPT
        if debug:
            print('get_response from claude api')
            print('messages: ', messages)
            print('prompt: ', prompt)
            print('context_info: ', context_info)
            print('temperature: ', temperature)
            print('model: ', model)

        claude_prompt = '\n\nHuman:'

        # if context_info:
        #     if isinstance(context_info, dict):
        #         context_info = json.dumps(context_info, ensure_ascii=False, indent=2)
        #     if isinstance(context_info, str):
        #         context_info = context_info.strip()
        #     if isinstance(context_info, list):  # messages
        #         for cntx_msg in context_info:
        #             claude_prompt += f'\n\n{claude_role(cntx_msg["role"])}: {cntx_msg["content"]}'
        #     else:
        #         claude_prompt += f'\n\n{context_info}'

        #     claude_prompt += '\n\nContext:' + context_info

        if context_info:
            claude_prompt += f'\n\n{context_info}\n\n' if context_info else ''

        if messages:
            claude_prompt += '\n\n[Previous messages]'
            for message in messages:
                claude_prompt += f'\n\n<{claude_role(message.role)}>: {message.content}'
        if prompt:
            claude_prompt += f'\n\n{prompt}\n\n' if prompt else ''
        claude_prompt += '\n\nAssistant:'

        request_obj = {
            'prompt': claude_prompt,
            'model': model,
            'temperature': temperature,
            'max_tokens_to_sample': max_tokens or 20000,
        }
        headers = {
            'accept': 'application/json',
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
            'x-api-key': api_key,
        }

        if debug:
            print('CLAUDE API REQUEST:')
            print(json.dumps(request_obj, indent=2, ensure_ascii=False))

        resp = requests.post(
            'https://api.anthropic.com/v1/complete',
            json=request_obj,
            headers=headers,
            timeout=60,
        ).json()

        if debug:
            print('CLAUDE API RESPONSE:')
            print(json.dumps(resp, indent=2, ensure_ascii=False))
        if 'error' in resp:
            print('Error in Claude API response: ', resp)
            raise Exception(resp[f'{resp["type"]}: {resp["message"]}'])
        return resp['completion']
