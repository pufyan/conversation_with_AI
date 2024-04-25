# encoding: utf-8

# pip install openai numpy sounddevice soundfile asyncio python-dotenv

import multiprocessing
import threading
import os
import time
from uuid import uuid1
import sounddevice as sd
import soundfile as sf
import json
from openai import OpenAI
import numpy as np
import shutil
#from utils.bot_utils import log
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import datetime
import pyaudio
import wave
load_dotenv()

INPUT_DEVICE = 1
OUTPUT_DEVICE = 3
SILENCE_THRESHOLD = 0.08
SILENCE_DURATION = 1.5
RECORD_DURATION = 6

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# remove audio folder if exists, with all files inside:
if os.path.exists('audio'):
    shutil.copytree('audio', f'audio_{uuid1()}')
    shutil.rmtree('audio')
os.makedirs('audio', exist_ok=True)

#import asyncio


#def sync_log(*args):
    # call async log function from bot_utils.py
#    asyncio.run(log('DEBUG', *args))


def is_silence(audio_chunk):
    """Проверяет, превышает ли амплитуда пороговое значение."""

    # print(np.max(np.abs(audio_chunk)))
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
                    print("Завершение записи по тишине.")
                    # Тишина длится более SILENCE_DURATION секунд
                    # audio_queue.put_nowait('STOP')
                    break
            else:
                silence_start_time = None

            if (time.time() - start_time >= RECORD_DURATION) and silence:
                stream.stop()
                print("Завершение записи по времени.")
                break
    # Сохранение записи в файл

    stream.close()
    concatenated_audio = np.concatenate(audio_buffer, axis=0)
    sf.write(filename, concatenated_audio, fs)
    return True
'''
def record_anything(filename):
    audio_format = pyaudio.paInt16
    channels = 2
    fs = 48000
    chunk_size = 2400  # размер фрагмента для чтения
    audio_buffer = []

    p = pyaudio.PyAudio()
    stream = p.open(format=audio_format, channels=channels, rate=fs, input=True, frames_per_buffer=chunk_size)

    recording = False
    silence_start_time = None
    start_time = None

    try:
        while True:
            audio_chunk = np.fromstring(stream.read(chunk_size), dtype=np.int16)
            audio_buffer.append(audio_chunk)

            silence = is_silence(audio_chunk)
            current_time = time.time()

            if not recording and not silence:
                recording = True
                print(f'Начинаю запись {filename}')
                start_time = current_time

            elif recording:
                if silence:
                    if silence_start_time is None:
                        silence_start_time = current_time
                    elif current_time - silence_start_time > SILENCE_DURATION:
                        print("Завершение записи по тишине.")
                        break
                else:
                    silence_start_time = None

                if (current_time - start_time >= RECORD_DURATION) and silence:
                    print("Завершение записи по времени.")
                    break

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

        concatenated_audio = np.concatenate(audio_buffer)
        wave_file = wave.open(filename, 'wb')
        wave_file.setnchannels(channels)
        wave_file.setsampwidth(p.get_sample_size(audio_format))
        wave_file.setframerate(fs)
        wave_file.writeframes(concatenated_audio.tobytes())
        wave_file.close()
'''
def record_audio(recordings_queue, allow_recording):
    print('Process started: RECORD')
    # Set up PyAudio to record
    rec_number = 0
    while True:
        print('Start recording...')
        filename = os.path.join('audio', f"audio_{uuid1()}.wav")
        if record_anything(filename):
            recordings_queue.put((filename, rec_number, allow_recording.value))
            print('Adding to queue', filename)
            rec_number += 1


def transcribe_audio(recordings_queue, texts_queue, text_to_ai_queue, allow_recording, count_transcribe_file):
    print('Process started: TRANSCRIBE')

    while True:
        if not recordings_queue.empty():
            filename, rec_number, allow_put = recordings_queue.get()
            thread = threading.Thread(target=thread_transcribe, args=(
            filename, rec_number, allow_put, texts_queue, text_to_ai_queue, allow_recording, count_transcribe_file))
            thread.start()


# Тут запускаем новый поток
def thread_transcribe(filename, rec_number, allow_put, texts_queue, text_to_ai_queue, allow_recording, count_transcribe_file):
    print(f'Получил для транскрибации {filename}')
 #   sync_log(f'Получил для транскрибации {filename}')
   
   # try:
    
    model = WhisperModel("base", download_root='faster_whisper_cache')
    transcript_text = ''
    with open(filename, "rb") as audio_file:
        segments, info = model.transcribe(audio_file)
        for segment in segments:
                # print( "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            print(f'{datetime.datetime.utcnow().strftime("%H:%M:%S,%f")}:[{segment.start:.2f} -> {segment.end:.2f}] {segment.text}')
            transcript_text += ' ' + segment.text
    del model           
    '''
    with open(filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                language='ru',
                model="whisper-1",
                temperature=0.2
                #timeout=13
            )
    transcript_text = transcript
        
    '''
    '''    
    except:
        data, fs = sf.read("SpeechMisrecognition.wav", dtype='float32')
        sd.play(data, fs, device=OUTPUT_DEVICE)
        count_transcribe_file.value += 1
        print('Потерял!')
        return 
    ''' 
    is_bad = False

    for bad_text in [
        "Продолжение следует",
        "Игорь Негода",
        "субтитры",
        "Спасибо за внимание!"
        "Субтитры",
        "Редактор субтитров",
        "Вот и всё. Спасибо за внимание. До новых встреч. ",
        "До новых встреч!",
        "До встречи!",
        "Будьте здоровы",
        "Всем пока",
        "И не забывайте подписаться на канал",
        "ПОДПИШИСЬ!",
        "ПОДПИСЫВАЙТЕСЬ НА КАНАЛ",
        "Удачи!",
        "😎",
        "С вами был я, Сергей Трофимов.",
        " \n",
        "🤣",
        "Пока!",
        "Пока-пока!",
        "Пока! Пока! Пока!",
        "С вами был Иван Акцинь",
        "DimaTorzok",
        "До свидания!",
        "Увидимся!",
        "Корректор А.Кулакова",
        "Подпишись на канал, ставь лайк и жми на колокольчик.Подпишись на канал, ставь лайк и жми на колокольчик",
        "Весьма спасибо за просмотр!"
    ]:
        if bad_text in transcript_text:
            print('BAD TEXT', transcript_text)
            is_bad = True

    if is_bad:
        count_transcribe_file.value += 1
        return

    count_transcribe_file.value += 1

    while allow_recording.value:
        if (rec_number <= count_transcribe_file.value) and allow_put:
            text_to_ai_queue.put(transcript_text)
            print(f'добавил: {transcript_text}')
            words_for_answer = ["Ответь", "ответь", "ОТВЕТЬ", "отвечай", "Отвечай"]
            if any(word in transcript_text for word in words_for_answer):
  #              sync_log(f'Получил текст для ответа:\n{transcript.texti}')
                text_to_ai = ""
                while not text_to_ai_queue.empty():
                    text_to_ai += text_to_ai_queue.get() + " "

                texts_queue.put(text_to_ai)
                print(f"Послал текст: {text_to_ai}")
                data, fs = sf.read("SpeechOn.wav", dtype='float32')
                sd.play(data, fs, device=OUTPUT_DEVICE)
                #sd.wait()
    #        else:
   #             sync_log(f'Плохой текст для ответа:\n{transcript.tex}')
            break
        
    else:
        print('Не отправляю!')
        while not texts_queue.empty():
            texts_queue.get()

    count_transcribe_file.value += 1
    print('transcribed', transcript_text)

PROMPT = '''Ты Голосовой ассистент в групповом чате. 
Отвечай максимально коротко, не больше 2-3 предолжений.
Отвечай только на русском языке.
О себе говори только в женском роде.
Учитывай что текст может быть не полным или поступать с задержкой - есть контекст не до конца понятный, подожди следующего сообщения или уточни вопрос.
'''


def get_answer_ai(text_to_send, messages):
    # return 'This is a test answer'
    print('----\nQuestion:\n', text_to_send)
  #  sync_log('Запрос:\n', text_to_send)
    messages = [
        {
            'content': PROMPT,
            'role': 'system'},
    ] + messages + [
        {'content': text_to_send, 'role': 'user'}
    ]
    print('----\nMessages:\n', json.dumps(messages, indent=4, ensure_ascii=False))
    response = client.chat.completions.create(
        messages=messages,
        model='gpt-3.5-turbo',
        # model='gpt-4-1106-preview'
    )
    answer = response.choices[0].message.content
    print('----\nAnswer:\n', answer)
    print('----')
    return answer


def process_text(texts_queue, answers_queue, allow_recording):
    messages = []
    print('Process started: PROCESS')
    while True:
        if allow_recording.value and (not texts_queue.empty()):
            text = texts_queue.get()
            if text:
                answer = get_answer_ai(text, messages)
                messages.append({'content': text, 'role': 'user'})
                messages.append({'content': answer, 'role': 'assistant'})

                answers_queue.put(answer)
                while not texts_queue.empty():
                    texts_queue.get()


def play(text):
    # Преобразование текстового ответа в речь
    audio_response = client.audio.speech.create(
        model="tts-1",
        input=text,
        # voice="alloy",  # Вы можете выбрать другие голоса
        voice="nova",
        # voice='onyx',
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

                play(text)

                data, fs = sf.read("SpeechOff.wav", dtype='float32')
                sd.play(data, fs, device=OUTPUT_DEVICE)
                sd.wait()

                print('Playing finished', text)
                time.sleep(5)
                while not texts_queue.empty():
                    text_to_ai_queue.get()
                    texts_queue.get()
                allow_recording.value = True
                print(allow_recording.value)



if __name__ == "__main__":

    recordings_queue = multiprocessing.Queue()
    texts_queue = multiprocessing.Queue()
    text_to_ai_queue = multiprocessing.Queue()
    answers_queue = multiprocessing.Queue()
    allow_recording = multiprocessing.Value('i', True, lock=True)
    count_transcribe_file = multiprocessing.Value('i', 0)

    processes = [

        multiprocessing.Process(target=record_audio, args=(recordings_queue, allow_recording)),
        multiprocessing.Process(target=transcribe_audio,
                                args=(recordings_queue, texts_queue, text_to_ai_queue, allow_recording, count_transcribe_file)),
        multiprocessing.Process(target=process_text, args=(texts_queue, answers_queue, allow_recording)),
        multiprocessing.Process(target=voice_text, args=(answers_queue, allow_recording, texts_queue)),
    ]
    for p in processes:
        p.start()
        time.sleep(0.3)

    for p in processes:
        p.join()


