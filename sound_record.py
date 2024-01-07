import sounddevice as sd
import soundfile as sf
import asyncio
import os
from asyncio import Queue
import numpy as np
from openai import OpenAI
import time
from uuid import uuid1
from dotenv import load_dotenv

load_dotenv()

INPUT_DEVICE = 1
OUTPUT_DEVICE = 3

SILENCE_THRESHOLD = 0.07
SILENCE_DURATION = 1.5
RECORD_DURATION = 5

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

recordings_queue = Queue()
texts_queue = Queue()

allow_recording = True
count_transcribe_file = 0


def is_silence(audio_chunk):
    """Проверяет, превышает ли амплитуда пороговое значение."""
    #print(np.max(np.abs(audio_chunk)))
    return np.max(np.abs(audio_chunk)) < SILENCE_THRESHOLD

def record_anything(filename):
    #print('recording...')

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
                    # Тишина длится более SILENCE_DURATION секунд
                    # audio_queue.put_nowait('STOP')
                    break
            else:
                silence_start_time = None

            if (time.time() - start_time >= RECORD_DURATION) and silence:
                break
    # Сохранение записи в файл
    stream.stop()
    stream.close()
    concatenated_audio = np.concatenate(audio_buffer, axis=0)
    sf.write(filename, concatenated_audio, fs)
    return True

def record_audio(recordings_queue):
    print('Process started: RECORD')

    rec_number = 0
    while True:
        print('Start recording...')
        filename = os.path.join('audio', f"audio_{uuid1()}.wav")
        if record_anything(filename):
            print('Adding to queue', filename)
            recordings_queue.put_nowait((filename, rec_number))
            rec_number += 1
async def async_record_audio(recordings_queue):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: record_audio(recordings_queue))

async def transcribe_audio(recordings_queue, texts_queue, allow_recording):
    print('Process started: TRANSCRIBE')
    count_transcribe_file = 0

    async def async_trans_audio(filename, rec_number):
        print('wwwwww')
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: async_trans(filename, rec_number))
    def async_trans(filename, rec_number):
        nonlocal count_transcribe_file
        print('Транс')
        with open(filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                language='ru',
                model="whisper-1"
            )

        is_bad = False

        for bad_text in [
            "Продолжение следует...",
            "Игорь Негода",
            "субтитры",
            "Субтитры",
            "Редактор субтитров",
            "До новых встреч!",
            "До встречи!",
            "Будьте здоровы",
            "Всем пока!",
            "И не забывайте подписаться на канал",
            "ПОДПИШИСЬ!",
            "ПОДПИСЫВАЙТЕСЬ НА КАНАЛ",
        ]:
            if bad_text in transcript.text:
                print('BAD TEXT', transcript.text)
                is_bad = True

        if is_bad:
            count_transcribe_file += 1
            return

        while allow_recording:
            if rec_number == count_transcribe_file:
                texts_queue.put_nowait(transcript.text)
                break
        else:
            print('Не отправляю!')
            while not texts_queue.empty():
                texts_queue.get_nowait()

        count_transcribe_file += 1

    while True:
        if not recordings_queue.empty():
            filename, rec_number = await recordings_queue.get()
            await async_trans_audio(filename, rec_number)
            print('qqqqqq')

def play(text):
    # Преобразование текстового ответа в речь
    audio_response = client.audio.speech.create(
        model="tts-1",
        input=text,
        # voice="alloy",  # Вы можете выбрать другие голоса
        voice="nova",
        response_format="opus"  # Формат аудиофайла
    )

    audio_response.stream_to_file("response.opus")
    data, fs = sf.read("response.opus", dtype='float32')
    sd.play(data, fs, device=OUTPUT_DEVICE)
    sd.wait()
    os.remove("response.opus")


async def get_answer_ai(text_to_send):
    # return 'This is a test answer'
    messages = [{'content': text_to_send, 'role': 'user'}]
    response = await client.chat.completions.create(
        messages=messages,
        model='gpt-3.5-turbo',
        # model='gpt-4-1106-preview'
    )
    answer = response.choices[0].message.content
    print('answer: ', answer)
    return answer


async def main():
    record_task = asyncio.create_task(async_record_audio(recordings_queue))
    transcribe_task = asyncio.create_task(transcribe_audio(recordings_queue, texts_queue, allow_recording))

asyncio.run(main())