# encoding: utf-8

# pip install openai numpy sounddevice soundfile asyncio python-dotenv

import multiprocessing
import wave
import os
import time
import queue
from uuid import uuid1
import sounddevice as sd
import soundfile as sf
import asyncio
from openai import OpenAI
import numpy as np
import shutil

INPUT_DEVICE = 1
OUTPUT_DEVICE = 3
SILENCE_THRESHOLD = 0.1
SILENCE_DURATION = 0.5
RECORD_DURATION = 5

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# remove audio folder if exists, with all files inside:
if os.path.exists('audio'):
    shutil.rmtree('audio')
os.makedirs('audio', exist_ok=True)


def is_silence(audio_chunk):
    """Проверяет, превышает ли амплитуда пороговое значение."""
    #print(np.max(np.abs(audio_chunk)))
    return np.max(np.abs(audio_chunk)) < SILENCE_THRESHOLD


def record_anything(filename):
    # print('recording...')

    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_buffer.append(indata.copy())

    channels = 2
    fs = 48000
    audio_buffer = []
    recording = False
    silence_start_time = None
    start_time = None

    stream = sd.InputStream(samplerate=fs, channels=channels, callback=callback, device=INPUT_DEVICE)

    while True:

        # Чтение данных с микрофона
        audio_chunk = sd.rec(int(0.05 * fs), samplerate=fs, channels=channels, device=INPUT_DEVICE, blocking=False)
        time.sleep(0.05)
        # Определение тишины
        silence = is_silence(audio_chunk)
        # Логика начала записи
        if not recording and not silence:
            recording = True
            stream.start()  # Добавление данных в буфер
            print(f'Начинаю запись {filename}')
            start_time = time.time()
        elif recording:
            # Логика завершения записи
            if silence:
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time > SILENCE_DURATION:
                    stream.stop()
                    # Тишина длится более SILENCE_DURATION секунд
                    # audio_queue.put_nowait('STOP')
                    break
            else:
                silence_start_time = None

            if (time.time() - start_time > RECORD_DURATION) and silence:
                stream.stop()
                break
    # Сохранение записи в файл

    stream.close()
    concatenated_audio = np.concatenate(audio_buffer, axis=0)
    sf.write(filename, concatenated_audio, fs)
    return True



def record_audio(recordings_queue):
    print('Process started: RECORD')
    # Set up PyAudio to record
    while True:
        #if allow_recording.value:
            print('Start recording...')
            filename = os.path.join('audio', f"audio_{uuid1()}.wav")
            if record_anything(filename):
                print('Adding to queue', filename)
                recordings_queue.put(filename)
            else:
                print('Silence')
                recordings_queue.put('STOP')
       # else:
            #print('Recording is not allowed!')
           # time.sleep(0.1)
        # print("recorded", filename)


def transcribe_audio(recordings_queue, texts_queue, allow_recording):
    print('Process started: TRANSCRIBE')
    current_text = ''
    while True:
        if not recordings_queue.empty():
            filename = recordings_queue.get()
            print(f'Забрал мз очереди {filename}')
            if filename == 'STOP':
                # print('STOP')
                # print(current_text)
                texts_queue.put(current_text)
                current_text = ''
                continue
            with open(filename, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    file=audio_file,
                    language='ru',
                    model="whisper-1"
                )

            for bad_text in [
                "продолжение следует...",
                'игорь негода',
                'субтитр',
                'Редактор субтитров А.Семкин Корректор А.Егорова',
            ]:
                if bad_text in transcript.text:
                    print('BAD TEXT', transcript.text)
                    continue

            current_text = transcript.text
            if allow_recording.value:
                texts_queue.put(current_text)
            else:
                print('Не отправляю!')
                allow_recording.value = True
            print('transcribed', transcript.text)


import json


def get_answer_ai(text_to_send, messages):
    # return 'This is a test answer'
    print('----\nQuestion:\n', text_to_send)
    messages = [
        {'content': 'Ты Голосовой ассистент. Отвечай максимально просто, коротко, только на русском языке', 'role': 'system'},
    ] + messages + [
        {'content': text_to_send, 'role': 'user'}
    ]
    print('----\nMessages:\n', json.dumps(messages, indent=4, ensure_ascii=False))
    response = client.chat.completions.create(
        messages=messages,
        # model='gpt-3.5-turbo',
        model='gpt-4-1106-preview'
    )
    answer = response.choices[0].message.content
    print('----\nAnswer:\n', answer)
    print('----')
    return answer


def process_text(texts_queue, answers_queue):
    messages = []
    print('Process started: PROCESS')
    while True:
        if not texts_queue.empty():
            text = texts_queue.get()
            print(text)
            if text:
                answer = get_answer_ai(text, messages)
                messages.append({'content': text, 'role': 'user'})
                messages.append({'content': answer, 'role': 'assistant'})
                answers_queue.put(answer)


def play(text):
    # Преобразование текстового ответа в речь
    audio_response = client.audio.speech.create(
        model="tts-1",
        input=text,
        # voice="alloy",  # Вы можете выбрать другие голоса
        # voice="nova",
        voice='onyx',
        response_format="opus"  # Формат аудиофайла
    )

    audio_response.stream_to_file("response.opus")
    data, fs = sf.read("response.opus", dtype='float32')
    sd.play(data, fs, device=OUTPUT_DEVICE)
    sd.wait()
    os.remove("response.opus")


def voice_text(answers_queue, allow_recording, texts_queue):
    print('Process started: VOICE')
    while True:
        if not answers_queue.empty():
            text = answers_queue.get()
            if text:
                allow_recording.value = False
                print('Playing started', text)
                #while not texts_queue.empty():
                #    texts_queue.get()

                play(text)
                #time.sleep(5)

                print('Playing finished', text)
                #allow_recording.value = True
            # print(text)
            # os.system(f'say "{text}"')


if __name__ == "__main__":
    recordings_queue = multiprocessing.Queue()
    texts_queue = multiprocessing.Queue()
    answers_queue = multiprocessing.Queue()
    allow_recording = multiprocessing.Value('i', True)

    processes = [
        multiprocessing.Process(target=record_audio, args=(recordings_queue,)),
        multiprocessing.Process(target=transcribe_audio, args=(recordings_queue, texts_queue, allow_recording)),
        multiprocessing.Process(target=process_text, args=(texts_queue, answers_queue)),
        multiprocessing.Process(target=voice_text, args=(answers_queue, allow_recording, texts_queue)),
    ]
    for p in processes:
        p.start()
        time.sleep(0.1)
