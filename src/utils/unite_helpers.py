import re
import unicodedata
import pandas as pd
from dotenv import load_dotenv
import os

# Access environment variables
load_dotenv()
PATH_SOURCE_CSV = os.getenv("PATH_SOURCE_CSV")

class Helpers:
    
    def __init__(self):
        self.path = PATH_SOURCE_CSV
    
    # VietnameseToneNormalization.md
    # https://github.com/VinAIResearch/BARTpho/blob/main/VietnameseToneNormalization.md

    TONE_NORM_VI = {
        'òa': 'oà', 'Òa': 'Oà', 'ÒA': 'OÀ',\
        'óa': 'oá', 'Óa': 'Oá', 'ÓA': 'OÁ',\
        'ỏa': 'oả', 'Ỏa': 'Oả', 'ỎA': 'OẢ',\
        'õa': 'oã', 'Õa': 'Oã', 'ÕA': 'OÃ',\
        'ọa': 'oạ', 'Ọa': 'Oạ', 'ỌA': 'OẠ',\
        'òe': 'oè', 'Òe': 'Oè', 'ÒE': 'OÈ',\
        'óe': 'oé', 'Óe': 'Oé', 'ÓE': 'OÉ',\
        'ỏe': 'oẻ', 'Ỏe': 'Oẻ', 'ỎE': 'OẺ',\
        'õe': 'oẽ', 'Õe': 'Oẽ', 'ÕE': 'OẼ',\
        'ọe': 'oẹ', 'Ọe': 'Oẹ', 'ỌE': 'OẸ',\
        'ùy': 'uỳ', 'Ùy': 'Uỳ', 'ÙY': 'UỲ',\
        'úy': 'uý', 'Úy': 'Uý', 'ÚY': 'UÝ',\
        'ủy': 'uỷ', 'Ủy': 'Uỷ', 'ỦY': 'UỶ',\
        'ũy': 'uỹ', 'Ũy': 'Uỹ', 'ŨY': 'UỸ',\
        'ụy': 'uỵ', 'Ụy': 'Uỵ', 'ỤY': 'UỴ'
        }

    def normalize_vnese(self, text):
        for i, j in self.TONE_NORM_VI.items():
            text = text.replace(i, j)
        # Remove control characters (ASCII 0–31, plus DEL 127)
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        # normalize spacing
        text = text.replace('\xa0', ' ')
        # Normalize input text to NFC
        text = unicodedata.normalize("NFC", text)
        return text
    
    def read_csv_unite(self, path: str='') -> pd.DataFrame:
        
        if len(path) <= 0:
            path = self.path
        
        names = ["Tên dự án", "Phường", "Hệ thống", "Liên hệ", "Tên hệ thống", "Dạng phòng", "Số lượng phòng", "Phòng đang trống", "Tiến độ cập nhật", "Thông tin thiếu", "Phân công check phòng", "Call/Nhắn tin Zalo", "Đặc điểm", "Lan Anh", "Chính sách", "Nhóm zalo", "Link UNC", "Rổ hàng ONLINE", "Liệt kê driver hình"]
        df = pd.read_csv(path, index_col=0, encoding='utf-8-sig', names=names)
        
        # Fill NaN with 'Chưa có thông tin'
        df.fillna('Chưa có thông tin', inplace=True)
        df = df.iloc[2:].astype(str).reset_index(drop=True)
        
        return df
    
    def create_sentences_list(self, df : pd.DataFrame) -> list[str]:
        sentences = []
        for idx, row in df.iterrows():
            parts = []
            for col in df.columns:
                parts.append(f"{col}: {row[col]}")
            sentence = " | ".join(parts)
            sentence = self.normalize_vnese(sentence.replace('\n', ' '))
            sentences.append(sentence)
        return sentences

class ProcessCSV:
    
    def __init__(self):
        self.path = PATH_SOURCE_CSV

    
    def processing_context_csv(self, path: str = '') -> list[str]:
        
        """Return a list constain combine content.
        
        Output: src/data/csv/sequences.csv
                
        Param:
            + path: Relative path string.
        """
        
        if len(path) <= 0:
            path = self.path
        
        helper = Helpers()
        result = []
        
        df = helper.read_csv_unite(path)
        sentences_list = helper.create_sentences_list(df)
        sentences_list = [' '.join(arr.split()) for arr in sentences_list]

        # Sort sentences by content length
        sentences_list = sorted(sentences_list, key=len)
        df = pd.DataFrame({
            'ID': list(range(len(sentences_list))),
            'content': sentences_list
        })

        result = [f"ID: {row[0]} | {row[1]}" for row in df.values]
        
        return result
    
    # TODO: add function return dataframe with cloumns: col1, col2, .., combine_info 
