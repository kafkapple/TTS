import os
import random
import shutil
from pydub import AudioSegment
from tqdm import tqdm

def prepare_vits_data(input_folder, output_folder, train_ratio=0.9, sample_rate=22050, min_file_size_kb=50):
    # 출력 폴더 생성
    wavs_folder = os.path.join(output_folder, 'wavs')
    filelists_folder = os.path.join(output_folder, 'filelists')
    os.makedirs(wavs_folder, exist_ok=True)
    os.makedirs(filelists_folder, exist_ok=True)

    # 입력 폴더에서 mp3와 txt 파일 목록 가져오기
    files = os.listdir(input_folder)
    mp3_files = [f for f in files if f.endswith('.mp3')]
    txt_files = [f for f in files if f.endswith('.txt')]

    print(f"총 발견된 파일: MP3 {len(mp3_files)}개, TXT {len(txt_files)}개")

    # 파일 쌍 생성 및 검증
    paired_files = []
    excluded_files = []
    
    for mp3_file in tqdm(mp3_files, desc="처리 중인 파일", total=len(mp3_files)):
        base_name = os.path.splitext(mp3_file)[0]
        txt_file = base_name + '.txt'
        mp3_path = os.path.join(input_folder, mp3_file)
        
        if txt_file in txt_files:
            try:
                # MP3 파일만 크기 확인 (KB)
                if mp3_file.lower().endswith('.mp3'):
                    file_size_kb = os.path.getsize(mp3_path) / 1024
                    if file_size_kb < min_file_size_kb:
                        excluded_files.append((mp3_file, f"MP3 파일 크기가 너무 작음 ({file_size_kb:.2f}KB)"))
                        continue
                
                # MP3 파일 유효성 검사
                AudioSegment.from_mp3(mp3_path)
                
                # 텍스트 파일 내용 확인 (빈 파일만 체크)
                txt_path = os.path.join(input_folder, txt_file)
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if not text:
                        excluded_files.append((mp3_file, "텍스트 파일이 비어있음"))
                        continue
                
                paired_files.append((mp3_file, txt_file))
                
            except Exception as e:
                excluded_files.append((mp3_file, f"오류 발생: {str(e)}"))
                continue
    
    print(f"\n유효한 파일 쌍: {len(paired_files)}개")
    print(f"제외된 파일: {len(excluded_files)}개")
    
    if excluded_files:
        print("\n제외된 파일 목록:")
        for file, reason in excluded_files:
            print(f"- {file}: {reason}")

    # 데이터를 랜덤 셔플 후 분할
    random.shuffle(paired_files)
    train_split = int(len(paired_files) * train_ratio)
    train_files = paired_files[:train_split]
    val_files = paired_files[train_split:]

    # train.txt와 val.txt 생성
    with open(os.path.join(filelists_folder, 'train.txt'), 'w', encoding='utf-8') as train_f:
        for mp3_file, txt_file in tqdm(train_files, desc="처리 중인 파일", total=len(train_files)):
            # mp3를 wav로 변환하여 wavs 폴더에 저장
            base_name = os.path.splitext(mp3_file)[0]
            mp3_path = os.path.join(input_folder, mp3_file)
            wav_path = os.path.join(wavs_folder, base_name + '.wav')

            audio = AudioSegment.from_mp3(mp3_path)
            audio = audio.set_frame_rate(sample_rate)
            audio.export(wav_path, format='wav')

            # 텍스트 읽기
            with open(os.path.join(input_folder, txt_file), 'r', encoding='utf-8') as txt_f:
                text = txt_f.read().strip()

            # train.txt에 기록
            train_f.write(f"wavs/{base_name}.wav|{text}\n")

    with open(os.path.join(filelists_folder, 'val.txt'), 'w', encoding='utf-8') as val_f:
        for mp3_file, txt_file in tqdm(val_files, desc="처리 중인 파일", total=len(val_files)):
            # mp3를 wav로 변환하여 wavs 폴더에 저장
            base_name = os.path.splitext(mp3_file)[0]
            mp3_path = os.path.join(input_folder, mp3_file)
            wav_path = os.path.join(wavs_folder, base_name + '.wav')

            audio = AudioSegment.from_mp3(mp3_path)
            audio = audio.set_frame_rate(sample_rate)
            audio.export(wav_path, format='wav')

            # 텍스트 읽기
            with open(os.path.join(input_folder, txt_file), 'r', encoding='utf-8') as txt_f:
                text = txt_f.read().strip()

            # val.txt에 기록
            val_f.write(f"wavs/{base_name}.wav|{text}\n")

base_path = os.path.dirname(os.path.abspath(__file__))
processed_path = os.path.join(base_path, 'data', 'processed')
os.makedirs(processed_path, exist_ok=True)
prepare_vits_data(
    input_folder= os.path.join(base_path, 'data', 'splitted'),
    output_folder=processed_path,
    train_ratio=0.9,  # 원하는 비율로 수정 가능
    sample_rate=22050,  # 필요한 경우 샘플링 레이트 변경 가능
    min_file_size_kb=20  # 최소 파일 크기 설정 (KB)
)