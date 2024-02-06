import logging
from datetime import datetime
import functools


import random
import string
import os

ENABLE_LOGGING = True


def random_string(length):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


class MetricWriter:

    def __init__(self, logs_folder=None):

        # self.logs_folder = os.path.join('logs', random_string(8))
        self.logs_folder = logs_folder or 'logs'
        os.makedirs(self.logs_folder, exist_ok=True)

    def log_metrics(
        self,
        # basic
        created=None,
        model=None,
        method=None,

        # common
        time_elapsed=None,
        token_count=None,

        # tts, whisper
        file_name=None,
        file_duration=None,
        file_size=None,

        # llm
        input_text=None,
        output_text=None,
        input_tokens_llm=None,
        output_tokens_llm=None,
    ):
        if not ENABLE_LOGGING:
            return
        # save in json, only not none:

        log_filename = os.path.join(self.logs_folder, f'{model}_{method}_{random_string(8)}.json')

        log_dict = {
            'created': created or datetime.utcnow().isoformat(),
            'model': model,
            'method': method,
            'time_elapsed': time_elapsed,
            'token_count': token_count,
            'file_name': file_name,
            'file_duration': file_duration,
            'file_size': file_size,
            'input_text': input_text,
            'output_text': output_text,
            'input_tokens_llm': input_tokens_llm,
            'output_tokens_llm': output_tokens_llm,
        }

        log_dict = {k: v for k, v in log_dict.items() if v is not None}

        import json
        with open(log_filename, 'w', encoding='utf-8') as f:
            json.dump(log_dict, f, indent=4)


def log_errors(func):
    """Decorator to catch and log exceptions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in function {func.__name__!r}: {str(e)}")
            raise
    return wrapper
