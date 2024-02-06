

try:
    from app.utils.audio_utils import send_voice
except ImportError:
    print('TODO import send_voice from audio_utils')

import telegram
import os
import subprocess

from dotenv import load_dotenv
load_dotenv()

# Chat(active_usernames=('aihotline',), api_kwargs={'accent_color_id': 2}, id=-1002108445649, invite_link='https://t.me/+GgkcbPbxprQyMmEy', join_to_send_messages=True, permissions=ChatPermissions(api_kwargs={'can_send_media_messages': True}, can_add_web_page_previews=True, can_change_info=False, can_invite_users=True, can_manage_topics=True, can_pin_messages=False, can_send_audios=True, can_send_documents=True, can_send_messages=True, can_send_other_messages=True, can_send_photos=True, can_send_polls=True, can_send_video_notes=True, can_send_videos=True, can_send_voice_notes=True), photo=ChatPhoto(big_file_id='AQADAgAD2NgxGxoyoEoACAMAAy-crq0W____plkLAQVI-Sk0BA', big_file_unique_id='AQAD2NgxGxoyoEoB', small_file_id='AQADAgAD2NgxGxoyoEoACAIAAy-crq0W____plkLAQVI-Sk0BA', small_file_unique_id='AQAD2NgxGxoyoEoAAQ'), title='🔥☎️ Hotline AI | Горячая линия ИИ', type=<ChatType.SUPERGROUP>, username='aihotline')

BOT_TOKEN = os.getenv('BOT_TOKEN') or None
DEBUG_CHAT = os.getenv('DEBUG_CHAT') or None
VOICING = os.getenv('VOICING') or False

bot = telegram.Bot(BOT_TOKEN)


ESCAPE_SYMBOLS = [
    # '*', '`',
    '_',
    # '[', ']', '(', ')',
    '~', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!'
]

MD_MODE = 'MarkdownV2'
HTML_MODE = 'HTML'
# MDVALUE = 'Markdown'

TEST = '''*bold \*text*
_italic \*text_
__underline__
~strikethrough~
||spoiler||
*bold _italic bold ~italic bold strikethrough ||italic bold strikethrough spoiler||~ __underline italic bold___ bold*
[inline URL](http://www.example.com/)
[inline mention of a user](tg://user?id=123456789)
![👍](tg://emoji?id=5368324170671202286)
`inline fixed-width code`
```
pre-formatted fixed-width code block
```
```python
pre-formatted fixed-width code block written in the Python programming language
```'''


async def send_text_message(msg, chat_id, parse_mode=None, markdown=None, message_thread_id=None, **kwargs):
    if msg is None:
        return

    if markdown is not None:
        parse_mode = MD_MODE if markdown else parse_mode

    chunk = 4096
    for i in range(0, len(msg), chunk):
        try:
            _msg = msg[i:i + chunk]
            print(f'>> "{_msg[:10]}" -> "{chat_id}" (parse_mode: {parse_mode})')

            if parse_mode == MD_MODE:
                for s in ESCAPE_SYMBOLS:
                    _msg = _msg.replace(s, f'\\{s}')

            await bot.send_message(
                chat_id=chat_id,
                text=_msg,
                parse_mode=parse_mode,
                message_thread_id=message_thread_id,
                # **kwargs
            )
        except telegram.error.BadRequest as bad_request_error:
            # await log(
            #     'send_text_message',
            #     f'bad request error: {bad_request_error}, msg: "{msg[i:i + chunk][:100]}", chat_id: {chat_id}, markdown: {markdown}, kwargs: {kwargs}',
            #     markdown=False
            # )
            # fuck up with markdown parsing, remove it and send raw

            print(bad_request_error)
            _msg = msg[i:i + chunk]
            await bot.send_message(
                chat_id=chat_id,
                text=_msg,
                message_thread_id=message_thread_id,
                # **kwargs
            )


async def log(sender, text, markdown=False):
    # print(f'sending to debug chat: {sp.DEBUG_CHAT}')
    msg = f'[{sender}]: {text}'
    try:
        await send_text_message(msg, DEBUG_CHAT)
    except Exception as e:
        print('Error while logging')
        print('msg:', msg)
        # print('chat:', DEBUG_CHAT)
        print(e)


async def send_answer(msg, chat_id, voice=VOICING, buttons=None, parse_mode=None, message_thread_id=None, **kwargs):
    '''
    :param msg: message to send
    :param chat_id: chat id to send message to
    :param voice: whether to voice the message
    :param buttons: list of strings to use as buttons
    '''
    print('USING VOICING: ', voice)

    request_data = dict(
        msg=msg,
        chat_id=chat_id,
    )

    if buttons:
        request_data['reply_markup'] = telegram.ReplyKeyboardMarkup(
            keyboard=[
                [telegram.KeyboardButton(text=button) for button in buttons]
            ],
            resize_keyboard=True,
            one_time_keyboard=True
        )
    # await bot.send_message(
    await send_text_message(
        **request_data,
        parse_mode=parse_mode,
        message_thread_id=message_thread_id,
    )

    if voice:
        await send_voice(msg, chat_id)


async def save_voice(update, context):
    file_id = update.message.voice.file_id
    # Download voice message from Telegram servers:
    file = await context.bot.get_file(file_id)
    os.makedirs('voices', exist_ok=True)
    voice_path = os.path.join(
        'voices',
        f'{file_id}.ogg'
    )
    audio_file = await file.download_to_drive(voice_path)
    return audio_file


# Testing and retrieving chat id:

from telegram import Update
from telegram.ext import CommandHandler, CallbackContext
from telegram.ext import Application, CommandHandler


async def scan(update: Update, context: CallbackContext) -> None:
    # Get chat id
    chat_id = update.effective_chat.id

    # Get all messages in chat
    bot = context.bot
    chat = await bot.get_chat(chat_id)
    print(chat)
    # new_messages = await chat.get_all_members()
    # print(new_messages)


async def ping(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('pong')


def main() -> None:
    print('Starting bot listening for /scan command')
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("scan", scan))
    application.add_handler(CommandHandler("ping", ping))
    application.run_polling()


if __name__ == '__main__':
    main()
