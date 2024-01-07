import multiprocessing
import time
from audio_recorder import run_audio_recorder
from transcriber import run_transcriber
from voice_responder import run_voice_responder

def main():
    recordings_queue = multiprocessing.Queue()
    texts_queue = multiprocessing.Queue()
    answers_queue = multiprocessing.Queue()
    allow_recording = multiprocessing.Value('i', True)

    processes = [
        multiprocessing.Process(target=run_audio_recorder, args=(recordings_queue,)),
        multiprocessing.Process(target=run_transcriber, args=(recordings_queue, texts_queue, allow_recording)),
        multiprocessing.Process(target=run_voice_responder, args=(answers_queue, allow_recording))
        ]

    for process in processes:
        process.start()

        # Ожидание завершения процессов
    try:
        while True:
            time.sleep(0.1)
            # Здесь можно добавить дополнительную логику или проверки
    except KeyboardInterrupt:
        print("Завершение работы...")
        allow_recording.value = False  # Сигнал для остановки процессов
        for process in processes:
            process.join()  # Ожидание завершения каждого процесса
        print("Все процессы успешно завершены.")

    if __name__ == "__main__":
        main()