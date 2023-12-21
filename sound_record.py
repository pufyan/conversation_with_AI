import sounddevice as sd
import soundfile as sf
import asyncio
import os
from asyncio import Queue
import numpy as np
from openai import AsyncOpenAI
import time

#    0 Microsoft Sound Mapper - Input, MME (2 in, 0 out)
# >  1 CABLE Output (VB-Audio Virtual , MME (2 in, 0 out)
#    2 Microsoft Sound Mapper - Output, MME (0 in, 2 out)
# <  3 CABLE Input (VB-Audio Virtual C, MME (0 in, 2 out)
#    4 Primary Sound Capture Driver, Windows DirectSound (2 in, 0 out)
#    5 CABLE Output (VB-Audio Virtual Cable), Windows DirectSound (2 in, 0 out)
#    6 Primary Sound Driver, Windows DirectSound (0 in, 2 out)
#    7 CABLE Input (VB-Audio Virtual Cable), Windows DirectSound (0 in, 2 out)
#    8 CABLE Input (VB-Audio Virtual Cable), Windows WASAPI (0 in, 2 out)
#    9 CABLE Output (VB-Audio Virtual Cable), Windows WASAPI (2 in, 0 out)
#   10 CABLE Output (VB-Audio Point), Windows WDM-KS (8 in, 0 out)
#   11 Speakers (VB-Audio Point), Windows WDM-KS (0 in, 8 out)

INPUT_DEVICE = 1
OUTPUT_DEVICE = 3

SILENCE_THRESHOLD = 0.05
SILENCE_DURATION = 0.5
RECORD_DURATION = 4

api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

audio_queue = Queue()
#text_queue = Queue()

async def is_silence(audio_chunk):
    """Проверяет, превышает ли амплитуда пороговое значение."""
    print(np.max(np.abs(audio_chunk)))
    return np.max(np.abs(audio_chunk)) < SILENCE_THRESHOLD

async def record_audio(filename):
    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_buffer.append(indata.copy())

    # Подготовка для записи
    channels = 2
    fs = 48000
    recording = False
    silence_start_time = None
    start_time = None
    audio_buffer = []
    stream = sd.InputStream(samplerate=fs, channels=channels, callback=callback, device=INPUT_DEVICE)

    while True:
        # Чтение данных с микрофона
        audio_chunk = sd.rec(int(0.05 * fs), samplerate=fs, channels=channels, device=INPUT_DEVICE, blocking=False)
        await asyncio.sleep(0.05)
        # Определение тишины
        silence = await is_silence(audio_chunk)
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
                    #audio_queue.put_nowait('STOP')
                    break

            else:
                silence_start_time = None
            if time.time() - start_time > RECORD_DURATION:
                break
    # Сохранение записи в файл
    stream.stop()
    stream.close()
    concatenated_audio = np.concatenate(audio_buffer, axis=0)
    sf.write(filename, concatenated_audio, fs)
    audio_queue.put_nowait(filename)

async def transcribe_audio(queue):
    result = ''
    async def async_trans(filename):
        trans_start_time = time.time()
        nonlocal result

        if filename == 'STOP':
            #answer_text = await get_answer_ai(result)
            #print('answer_text: ', answer_text)
            #await play(answer_text)
            # with open(f"{filename}.txt", 'w') as file:
            #     file.write(result)

            result = ''
            return

        else:
            print(f"Начинаю транскрибацию файла {filename}.")
            with open(filename, "rb") as audio_file:
                transcript = await client.audio.transcriptions.create(file=audio_file, language='ru', model="whisper-1")
            print(transcript.text)
            result += transcript.text
            print(f"Транскрибация файла {filename} завершена за {time.time() - trans_start_time} сек.")
            os.remove(filename)  # Удаляем файл после транскрибации

        answer_text = await get_answer_ai(result)
        print('answer_text: ', answer_text)
        await play(answer_text)

        queue.task_done()

    while True:
        filename = await queue.get()
        asyncio.create_task(async_trans(filename))

async def continuous_recording():
    segment = 1
    while True:
        filename = f"audio_segment_{segment}.wav"
        await record_audio(filename)
        segment += 1


SYSTEM_PROMPT = ''''''


async def play(text):
    # Преобразование текстового ответа в речь
    audio_response = await client.audio.speech.create(
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
    transcription_task = asyncio.create_task(transcribe_audio(audio_queue))
    await continuous_recording()

asyncio.run(main())