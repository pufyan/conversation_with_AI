import os
import json
import traceback
from datetime import datetime


def read_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                try:
                    data.append(json.load(file))
                except json.JSONDecodeError as e:
                    print(f"Ошибка чтения {filename}", e, traceback.format_exc())
    return data


def calculate_report(data):
    total_requests = len(data)

    total_time = 0
    first_request_time = None
    last_request_time = None

    for d in data:
        if first_request_time is None:
            # "2024-02-01T17:32:14.091605"
            first_request_time = datetime.fromisoformat(d['created'])
            continue

        if last_request_time is None:
            last_request_time = datetime.fromisoformat(d['created'])
            continue

        dtvalue = datetime.fromisoformat(d['created'])

        if dtvalue < first_request_time:
            first_request_time = dtvalue
        elif dtvalue > last_request_time:
            last_request_time = dtvalue

    total_time = (last_request_time - first_request_time).total_seconds()

    total_input_tokens = sum(item['input_tokens_llm'] for item in data)
    total_output_tokens = sum(item['output_tokens_llm'] for item in data)

    total_words_input = sum(len(item['input_text'].split()) for item in data)
    total_words_output = sum(len(item['output_text'].split()) for item in data)

    total_chars_input = sum(len(item['input_text']) for item in data)
    total_chars_output = sum(len(item['output_text']) for item in data)

    avg_time = total_time / total_requests
    avg_input_tokens = total_input_tokens / total_requests
    avg_output_tokens = total_output_tokens / total_requests

    avg_words_input = total_words_input / total_requests
    avg_words_output = total_words_output / total_requests

    avg_chars_input = total_chars_input / total_requests
    avg_chars_output = total_chars_output / total_requests

    total_cost_input = total_input_tokens / 1000 * 0.01
    total_cost_output = total_output_tokens / 1000 * 0.03
    total_cost = total_cost_input + total_cost_output

    avg_cost_input = avg_input_tokens / 1000 * INPUT_COST_PER_1K_TOKENS
    avg_cost_output = avg_output_tokens / 1000 * OUTPUT_COST_PER_1K_TOKENS
    avg_cost_total = avg_cost_input + avg_cost_output

    avg_cost_per_100word_input = total_cost_input / total_words_input * 100
    avg_cost_per_100word_output = total_cost_output / total_words_output * 100
    avg_cost_per_100word_total = total_cost / (total_words_input + total_words_output) * 100

    avg_cost_per_1000char_input = total_cost_input / total_chars_input * 1000
    avg_cost_per_1000char_output = total_cost_output / total_chars_output * 1000
    avg_cost_per_1000char_total = total_cost / (total_chars_input + total_chars_output) * 1000

    report = f"""Модель: gpt-4-0125-preview
Количество запросов: {total_requests}

Среднее время запроса: {avg_time:.2f} сек
Продолжительность теста: {total_time / 60:.2f} мин

Среднее количество токенов в input за 1 запрос: {avg_input_tokens:.0f}
Среднее количество токенов в output за 1 запрос: {avg_output_tokens:.0f}
Среднее количество токенов в сумме за 1 запрос: {avg_input_tokens + avg_output_tokens:.0f}

Суммарно слов в input: {total_words_input}
Суммарно слов в output: {total_words_output}
Суммарно слов в input+output: {total_words_input + total_words_output}

Среднее количество слов в input за 1 запрос: {avg_words_input:.0f}
Среднее количество слов в output за 1 запрос: {avg_words_output:.0f}
Среднее количество слов в input+output за 1 запрос: {avg_words_input + avg_words_output:.0f}

Суммарно символов в input: {total_chars_input}
Суммарно символов в output: {total_chars_output}
Суммарно символов в input+output: {total_chars_input + total_chars_output}

Среднее количество символов в input за 1 запрос: {avg_chars_input:.0f}
Среднее количество символов в output за 1 запрос: {avg_chars_output:.0f}
Среднее количество символов в input+output за 1 запрос: {avg_chars_input + avg_chars_output:.0f}
---
Суммарная стоимость input ($): {total_cost_input:.2f}
Суммарная стоимость output ($): {total_cost_output:.2f}
Суммарная стоимость input+output ($): {total_cost:.2f}

Средняя стоимость input ($ за 1 запрос): {avg_cost_input:.5f}
Средняя стоимость output ($ за 1 запрос): {avg_cost_output:.5f}
Средняя стоимость input+output ($ за 1 запрос): {avg_cost_total:.5f}

Средняя стоимость input ($ за 100 слов): {avg_cost_per_100word_input:.5f}
Средняя стоимость output ($ за 100 слов): {avg_cost_per_100word_output:.5f}
Средняя стоимость input+output ($ за 100 слов): {avg_cost_per_100word_total:.5f}

Средняя стоимость input ($ за 1000 символов): {avg_cost_per_1000char_input:.5f}
Средняя стоимость output ($ за 1000 символов): {avg_cost_per_1000char_output:.5f}
Средняя стоимость input+output ($ за 1000 символов): {avg_cost_per_1000char_total:.5f}"""
    # Среднее количество токенов в input за слово: {avg_token_cost_per_word_input:.5f}
    # Среднее количество токенов в output за слово: {avg_token_cost_per_word_output:.5f}
    # Среднее количество токенов в input+output за слово: {avg_token_cost_per_word_total:.5f}
    # """
    return report


# gpt3
INPUT_COST_PER_1K_TOKENS = 0.0005  # 0.01
OUTPUT_COST_PER_1K_TOKENS = 0.0015  # 0.03

# gpt4
# INPUT_COST_PER_1K_TOKENS = 0.01
# OUTPUT_COST_PER_1K_TOKENS = 0.03

directory = "server_logs"
data = read_files(directory)
report = calculate_report(data)
print(report)
