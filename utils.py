import os

class Utils:
    @staticmethod
    def save_file(text: str, file_path: str) -> None:
        """
        텍스트를 파일로 저장합니다.
        
        Args:
            text (str): 저장할 텍스트
            file_path (str): 저장할 파일 경로
        """
        try:
            # 파일이 위치할 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 텍스트를 파일에 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
        except Exception as e:
            print(f"파일 저장 중 오류 발생: {str(e)}") 