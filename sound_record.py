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
audio_queue = Queue()
text_queue = Queue()
input_device = 2 # Указываем источник аудио
output_device = 4 # Указываем выход аудио
SILENCE_THRESHOLD = 0.01

api_key = ""
client = AsyncOpenAI(api_key=api_key)

async def is_silence(audio_chunk):
    """Проверяет, превышает ли амплитуда пороговое значение."""
    print(np.sqrt(np.mean(audio_chunk**2)))
    return np.sqrt(np.mean(audio_chunk**2)) < SILENCE_THRESHOLD


async def record_audio(filename, duration, fs=48000):
    print(f"Проверка наличия аудио потока...")
    while True:
        # Получаем небольшой фрагмент аудио для проверки
        test_chunk = sd.rec(int(fs * 0.1), samplerate=fs, channels=2, device=sound_device)
        await asyncio.sleep(0.1)  # Длительность фрагмента
        if not await is_silence(test_chunk):
            break
    print(f"Начало записи в файл {filename}.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=2, device=sound_device)
    await asyncio.sleep(duration)
    sf.write(filename, audio, fs)
    print(f"Запись в файл {filename} завершена.")
    await audio_queue.put(filename)
    print('размер очереди = ' + str(audio_queue.qsize()))


async def transcribe_audio(queue):
    while True:
        filename = await queue.get()
        print(f"Начинаю транскрибацию файла {filename}.")
        with open(filename, "rb") as audio_file:
            transcript = await client.audio.transcriptions.create(file=audio_file, language='ru', model="whisper-1")
        print(transcript.text)
        result = transcript.text

        if result != '':
            await get_answer_ai(result)
        print(f"Транскрибация файла {filename} завершена: {result}")
        os.remove(filename)  # Удаляем файл после транскрибации
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