import whisper
import os
import sys

from pydub import AudioSegment
from tqdm import tqdm
import os
from pathlib import Path

# path = Path("c:/bin")



# AudioSegment.converter = path

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from summarizer.utils import Utils
path  = os.path.dirname(os.path.abspath(__file__))
print('\n####',path)

file_path = os.path.join(path, 'data', 'shoco.mp3')
save_path = os.path.join(path, 'save')
os.makedirs(save_path, exist_ok=True)
print(file_path, save_path)
# 모델 로드
model = whisper.load_model("base")
# 음성 파일 로드 및 변환
audio = AudioSegment.from_file(file_path)
result = model.transcribe(file_path, language='ko', task='transcribe', word_timestamps=True)
# 결과 출력
for i, segment in tqdm(enumerate(result['segments'])):
    start_ms = segment['start'] * 1000  # 밀리초 변환
    end_ms = segment['end'] * 1000
    text = segment['text']    # 인식된 텍스트
    print(f"[{start_ms} - {end_ms}] {text}")
    audio_segment = audio[start_ms:end_ms]
    i_file = os.path.join(save_path, f"sentence_{i+1}_{segment['id']}")
    Utils.save_file(text, i_file+".txt" )
    audio_segment.export(i_file+".mp3", format="mp3")

