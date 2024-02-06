
import os
import torch
import datetime
import telegram
import sys
import torch


import subprocess

sys.path.append(os.getcwd())
from app.settings import sp


bot = telegram.Bot(sp.TOKEN)

VOICING = str(sp.VOICING).lower() in ['true', '1', 'yes', 'y']
if VOICING:
    print('init model: ', datetime.datetime.now().strftime("%H:%M:%S"))
    device = torch.device('cpu')
    torch.set_num_threads(4)
    local_file = 'model.pt'

    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file(
            'https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
            local_file
        )

    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)
    sample_rate = 48000
    # speaker = 'baya'

    # speaker = 'random'
    # aidar, baya, kseniya, xenia, random
    # en:
    # en_0, en_1, ..., en_117, random

    print('done: ', datetime.datetime.now().strftime("%H:%M:%S"))


import asyncio


def run_async(async_function, *a, **kw):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_function(
        *a, **kw
    ))


async def convert_to_mp3(audio_file):
    # Convert to mp3:
    output_audio_file = f'{audio_file}.mp3'
    # ffmpeg_cmd = f"ffmpeg -i {audio_file} -vn -acodec copy {audio_file}.wav"
    ffmpeg_cmd = f"ffmpeg -i {audio_file} -vn -ar 44100 -ac 2 -ab 192k -f mp3 {output_audio_file}"
    subprocess.run(ffmpeg_cmd.split(), check=True)
    print('audio_file:', output_audio_file)
    return output_audio_file


async def send_tts(text, chat_id, speaker='random'):
    start = datetime.datetime.now()
    audio_path = create_audio(text, speaker)
    done_tts = datetime.datetime.now()
    print('sending...', datetime.datetime.now().strftime("%H:%M:%S"))
    with open(audio_path, 'rb') as f:
        message = await bot.send_voice(
            chat_id=chat_id,
            voice=f,
            read_timeout=60,
            write_timeout=60,
            connect_timeout=60,
            pool_timeout=60,
        )
        print(message.voice)

    print('done sending: ', datetime.datetime.now().strftime("%H:%M:%S"))
    done_sending = datetime.datetime.now()
    await bot.send_message(
        chat_id=sp.DEBUG_CHAT,
        text='speaker: {}. tts time: {}, sending time: {}'.format(
            speaker,
            (done_tts - start).total_seconds(),
            (done_sending - done_tts).total_seconds(),
        )
    )
    await bot.send_voice(
        chat_id=sp.DEBUG_CHAT,
        voice=message.voice.file_id
    )

    # await bot.send_audio(
    #     chat_id=DEBUG_CHAT,
    #     audio=open(audio_path, 'rb'),
    #     title='Silero TTS'
    # )

    # await bot.send_message(
    #     chat_id=DEBUG_CHAT,
    #     text=msg[i:i + chunk],
    # )

speakers = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene']
import time


def append(s1, s2):
    if s1:
        return s1 + '. ' + s2
    else:
        return s2


async def send_voice(msg, chat_id=sp.DEBUG_CHAT):
    speaker = 'eugene'
    await bot.send_message(
        chat_id=sp.DEBUG_CHAT,
        text='speaker: {}'.format(speaker)
    )
    try:
        msg = msg.replace('\n\n', ' ').replace('\n', ' ')
        chunk = 1000
        current_message = ''
        for line in msg.split('.'):
            if len(append(current_message, line)) < chunk:
                current_message = append(current_message, line)
            else:
                print(current_message)
                await send_tts(current_message, chat_id, speaker=speaker)
                current_message = line

        if current_message:
            print(current_message)
            await send_tts(current_message, chat_id, speaker=speaker)
    except Exception as e:
        print(e)
        await bot.send_message(
            chat_id=sp.DEBUG_CHAT,
            text=f"VOICING ERROR: {str(e)}. Message: '{msg}'"
        )


def create_audio(text, speaker):

    audio_paths = model.save_wav(
        text=text,
        speaker=speaker,
        sample_rate=sample_rate
    )
    return audio_paths


def merge_audio_files(audio_paths):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    merged_audio_path = f"merged_audio_{timestamp}.mp3"
    temp_file_path = f"temp_{timestamp}.txt"

    # Create a temporary file with the list of files to be merged
    with open(temp_file_path, "w") as f:
        for path in audio_paths:
            f.write(f"file '{path}'\n")

    # Use ffmpeg to merge the audio files
    command = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", temp_file_path, "-c:a", "libmp3lame", "-q:a", "4", merged_audio_path]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Delete the temporary file
    os.remove(temp_file_path)

    return merged_audio_path


if __name__ == '__main__':

    msg = '''Поэкспериментировать с gpt-3.5-turbo— отличный способ узнать, на что способен API. После того, как у вас появится представление о том, чего вы хотите достичь, вы можете остановиться на той gpt-3.5-turboили иной модели и попытаться оптимизировать ее возможности.
Вы можете использовать инструмент сравнения GPT , который позволяет параллельно запускать разные модели для сравнения выходных данных, настроек и времени отклика, а затем загружать данные в электронную таблицу Excel'''
    # create_audio_example()
    # sys.exit(0)

#     msg = """Существует множество программ для графического дизайна, но некоторые из них являются особенно популярными среди профессионалов в этой области. Ниже я перечислю топ-10 программ для графического дизайна:

# 1. Adobe Photoshop - наиболее известный и распространенный графический редактор, используемый для создания и редактирования растровых изображений.

# 2. Adobe Illustrator - векторный редактор, используемый для создания логотипов, иллюстраций, карточек на дизайн и многое другое.

# 3. Adobe InDesign - программа для верстки и дизайна печатной продукции, такой как книги, журналы, буклеты, рекламные брошюры, пакеты и т.д.

# 4. Sketch - популярный векторный редактор, используемый для создания интерфейсов и веб-дизайна.

# 5. GIMP - свободно распространяемый графический редактор, используемый как альтернатива Photoshop.

# 6. CorelDRAW - векторный редактор, используемый для создания иллюстраций, логотипов, визитных карточек, баннеров и многого другого.

# 7. Adobe XD - платформа для дизайна пользовательских интерфейсов и опыта пользования приложения.

# 8. Canva - онлайн-инструмент для создания профессиональных дизайнов без опыта обработки и работе в Photoshop.

# 9. Figma - инструмент для дизайна интерфейсов и коллаборативной работы с командами.

# 10. Procreate - программное обеспечение исключительно для пользователей iPad, используемое для рисования, создания иллюстраций и работ со штриховками.

# Все эти программы имеют свои преимущества, но выбор конкретной зависит от индивидуальных потребностей и предпочтений.
# """

    run_async(send_voice, msg, sp.DEBUG_CHAT)

    # print(len(msg))
    # run_async(send_tts, msg, DEBUG_CHAT)
