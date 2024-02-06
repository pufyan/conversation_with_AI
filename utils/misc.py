import asyncio
import json
import hashlib
import os
from utils.color_printing import *


def async_crush_on_timeout(timeout):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout)
            except asyncio.TimeoutError:
                print_r(f"Timeout of {timeout} seconds exceeded, try again...")

                # recreate client:

                # import time
                # time.sleep(1)
                # call again with the same arguments:

                return await wrapper(*args, **kwargs)
        return wrapper
    return decorator


def generate_cache_key(*args, **kwargs):
    # Convert args to a tuple
    args_tuple = tuple(args)

    # Sort kwargs alphabetically by keys
    sorted_kwargs = sorted(kwargs.items())

    # Combine args and sorted kwargs
    combined_data = (args_tuple, sorted_kwargs)

    # Serialize the combined data to a JSON string
    serialized_data = json.dumps(combined_data, sort_keys=True)

    # Hash the serialized data using SHA-256
    hash_key = hashlib.sha256(serialized_data.encode()).hexdigest()

    return hash_key


def use_cached_results(func):
    # look for cached results of any function in json files in the cache folder, and return parsed json
    def wrapper(*args, **kwargs):
        folder = os.path.join('func_caches', func.__name__)
        os.makedirs(folder, exist_ok=True)

        # always generate same hash for same args and kwargs:

        args_hash = generate_cache_key(*args, **kwargs)
        # args_hash = ''.join(sorted(json.dumps(args, ensure_ascii=False, indent=2) + json.dumps(kwargs, ensure_ascii=False, indent=2)))
        # args_hash = f"{'_'.join(args)}"
        # if kwargs:
        #     args_hash += "_".join(['_'.join([str(k), str(v)]) for k, v in kwargs.items()])
        filename = os.path.join(folder, str(args_hash) + '.json')
        print('Cache filename:', filename)
        if os.path.exists(filename):
            print_g(f'using cached results for {func.__name__} and key "{args_hash}"')

            with open(filename, 'r', encoding='utf-8') as f:
                json_string = f.read()
                return json.loads(json_string)
        else:
            print_r(f'calculating {func.__name__} and key "{args_hash}"')
            result = func(*args, **kwargs)
            print_y(result)
            json_string = json.dumps(result, ensure_ascii=False, indent=2)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(json_string)
            return result

    return wrapper
