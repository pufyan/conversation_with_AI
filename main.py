import sounddevice as sd
import soundfile as sf
import whisper

filename = ''  # Путь к файлу, который нужно воспроизводить
output_device = 4
# Функция для воспроизведения аудиофайла
print(sd.query_devices())
def play_audio(file):
    data, fs = sf.read(file, dtype='float32')
    sd.play(data, fs, device=output_device)
    sd.wait()  # Ожидание окончания воспроизведения

# Бесконечный цикл для зацикленного воспроизведения
while True:
    #model = whisper.load_model('base')
    #tr = model.transcribe(filename, fp16=False)
    #print(tr['text'])
    play_audio(filename)