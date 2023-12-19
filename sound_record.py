import sounddevice as sd
import soundfile as sf
import asyncio
import whisper
import os
from asyncio import Queue
import numpy as np
from openai import AsyncOpenAI,OpenAI
from openai.types.audio import Transcription
import openai
import time

audio_queue = Queue()
text_queue = Queue()
input_device = 1 # Указываем источник аудио
output_device = 5 # Указываем выход аудио
SILENCE_THRESHOLD = 0.01

api_key = ""
client = AsyncOpenAI(api_key=api_key)

async def is_silence(audio_chunk):
    """Проверяет, превышает ли амплитуда пороговое значение."""
    return np.max(np.abs(audio_chunk)) < SILENCE_THRESHOLD


async def record_audio(filename, duration, fs=48000):
    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_buffer.append(indata.copy())

    # Подготовка для записи
    sound_device = 1
    channels = 2
    recording = False
    silence_start_time = None
    start_time = None
    audio_buffer = []
    stream = sd.InputStream(samplerate=fs, channels=channels, callback=callback)

    while True:
        # Чтение данных с микрофона
        audio_chunk = sd.rec(int(0.1 * fs), samplerate=fs, channels=channels, device=sound_device, blocking=False)
        await asyncio.sleep(0.1)
        # Определение тишины
        silence = await is_silence(audio_chunk)
        # Логика начала записи
        if not recording and not silence:
            recording = True
            stream.start()  # Добавление данных в буфер
            start_time = time.time()
            print(f'Начало записи куска {time.time() - start_time}')
        elif recording:
            # Логика завершения записи
            if silence:
                if silence_start_time is None:
                    silence_start_time = time.time()
                    print('начало тишины в записи')
                elif time.time() - silence_start_time > 2:
                    # Тишина длится более 2 секунд
                    print('Добавляем тишину')
                    break
            else:
                silence_start_time = None

            if time.time() - start_time > 5:
                print('Прошло 5 сек')
                break

    # Сохранение записи в файл
    stream.stop()
    stream.close()
    concatenated_audio = np.concatenate(audio_buffer, axis=0)
    sf.write(filename, concatenated_audio, fs)
    audio_queue.put_nowait(filename)


async def transcribe_audio(queue):
    while True:
        filename = await queue.get()
        print(f"Начинаю транскрибацию файла {filename}.")
        with open(filename, "rb") as audio_file:
            transcript = await client.audio.transcriptions.create(file=audio_file, language='ru', model="whisper-1")
        print(transcript.text)
        result = transcript.text

        if result != '':
            #await get_answer_ai(result)
            with open(f"{filename}.txt", 'w') as file:
                file.write(result)
        print(f"Транскрибация файла {filename} завершена: {result}")
        #os.remove(filename)  # Удаляем файл после транскрибации
        print('удалил')
        queue.task_done()

async def continuous_recording():
    segment = 1
    while True:
        filename = f"audio_segment_{segment}.wav"
        await record_audio(filename, 10)
        segment += 1


async def get_answer_ai(text_to_send):
    text_response = await client.completions.create(
        model="text-davinci-003",  # Или другая модель GPT
        prompt=text_to_send,  # Используем текст из файла как входной запрос
        max_tokens=500
    )
    answer_text = text_response.choices[0].text

    # Преобразование текстового ответа в речь
    audio_response = await client.audio.speech.create(
        model="tts-1",
        input=answer_text,
        voice="alloy",  # Вы можете выбрать другие голоса
        response_format="opus"  # Формат аудиофайла
    )
    with open("response.opus", "wb") as file:
        file.write(audio_response.read())

    data, fs = sf.read("response.opus", dtype='float32')
    sd.play(data, fs, device=output_device)
    sd.wait()
    os.remove("response.opus")

async def main():
    transcription_task = asyncio.create_task(transcribe_audio(audio_queue))
    await continuous_recording()

asyncio.run(main())