# abbr_processor.py
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

from underthesea import (
    text_normalize,
    word_tokenize as uts_word_tokenize,
    pos_tag,
)
from pyvi import ViTokenizer, ViPosTagger


class AbbreviationProcessor:
    """
    Một class duy nhất:
      1. Nạp / lưu từ điển viết tắt
      2. Mở rộng viết tắt
      3. Tokenize
      4. Loại stop-word
      5. Trích cụm từ (chunk)
    """

    # ------------------------------------------------------------------
    # Khởi tạo
    # ------------------------------------------------------------------
    def __init__(
        self,
        abbr_path: Optional[str | Path] = None,
        stopwords_path: Optional[str | Path] = None,
    ) -> None:
        self._abbr: Dict[str, str] = {}
        self._stopwords: List[str] = []
        self._re_abbr: Optional[re.Pattern] = None

        # nạp dữ liệu
        if abbr_path:
            self.load_abbr(abbr_path)
        if stopwords_path:
            self.load_stopwords(stopwords_path)

    # ------------------------------------------------------------------
    # 1. Quản lý từ điển viết tắt
    # ------------------------------------------------------------------
    def load_abbr(self, path: str | Path) -> "AbbreviationProcessor":
        """Nạp từ điển từ file JSON."""
        path = Path(path)
        if path.exists():
            with path.open(encoding="utf-8") as f:
                self._abbr = json.load(f)
        else:
            # File không tồn tại → build mặc định (hard-code)
            self._build_default_abbr()
        self._compile_abbr_regex()
        return self

    def save_abbr(self, path: str | Path) -> None:
        """Lưu từ điển hiện tại ra JSON."""
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self._abbr, f, ensure_ascii=False, indent=2)

    def _build_default_abbr(self) -> None:
        """Từ điển mặc định (rút gọn, bạn có thể ném toàn bộ ABBR cũ vào đây)."""
        self._abbr = {
            "ko": "không",
            "k": "không",
            "hok": "không",
            "hg": "không",
            "kh": "không",
            "đc": "được",
            "dc": "được",
            "đk": "được",
            "dk": "được",
            "ib": "inbox",
            "ship": "giao hàng",
            "freeship": "miễn phí vận chuyển",
            "cod": "giao hàng thu tiền",
            "sp": "sản phẩm",
            "sl": "số lượng",
            "bh": "bảo hành",
            "hh": "hết hàng",
            "fb": "facebook",
            "gg": "giảm giá",
            "km": "khuyến mãi",
            "ny": "người yêu",
            "ck": "chồng",
            "vk": "vợ",
            "mn": "mọi người",
            "ae": "anh em",
            "se": "sẽ",
            "lun": "luôn",
            "nhg": "nhưng",
            "j": "gì",
            "m": "mình",
            "t": "tao",
            "r": "rồi",
            "z": "vậy",
            "oy": "rồi",
            "p": "phải",
            "s": "sao",
            "^^": "cười",
            ":v": "cười",
            "=))": "cười lớn",
            "huhu": "khóc",
            "hic": "khóc",
            "zzz": "buồn ngủ",
            "24/7": "24 giờ 7 ngày",
            "t2": "thứ hai",
            "t3": "thứ ba",
            "t7": "thứ bảy",
            "cn": "chủ nhật",
            "hn": "hôm nay",
            "bg": "bây giờ",
            "stt": "trạng thái",
            "like": "thích",
            "share": "chia sẻ",
            "sub": "đăng ký",
            "fl": "theo dõi",
            "acc": "tài khoản",
            "mk": "mật khẩu",
            "link": "liên kết",
            "web": "trang web",
            "avt": "ảnh đại diện",
            "cover": "ảnh bìa",
            "sz": "size",
            "đh": "đơn hàng",
            "mdh": "mã đơn hàng",
            "vc": "phí vận chuyển",
            "tt": "thanh toán",
            "tb": "thông báo",
            "yc": "yêu cầu",
            "tl": "trả lời",
            "rep": "trả lời",
            "ks": "cảm ơn",
            "thank": "cảm ơn",
            "tks": "cảm ơn",
            "thks": "cảm ơn",
            "hnay": "hôm nay",
            "qua": "hôm qua",
            "mai": "ngày mai",
            "bh": "bao giờ",
            "mấy h": "mấy giờ",
            "add": "địa chỉ",
            "sđt": "số điện thoại",
            "tel": "điện thoại",
            "pass": "mật khẩu",
            "un": "tên đăng nhập",
            "id": "tên đăng nhập",
            "tk": "tài khoản",
            "url": "đường dẫn",
            "blog": "nhật ký",
            "post": "bài viết",
            "tag": "gắn thẻ",
            "cate": "thể loại",
            "ep": "tập phim",
            "chap": "chương",
            "vol": "tập",
            "season": "mùa",
            "part": "phần",
            "clip": "video ngắn",
            "mp3": "file âm thanh",
            "mp4": "file video",
            "pdf": "file PDF",
            "doc": "file Word",
            "excel": "file Excel",
            "xls": "file Excel",
            "ppt": "file PowerPoint",
            "txt": "file văn bản",
            "zip": "file nén",
            "rar": "file nén",
            "7z": "file nén",
            "folder": "thư mục",
            "directory": "thư mục",
            "path": "đường dẫn",
            "bio": "tiểu sử",
            "status": "trạng thái",
            "ex": "ví dụ",
            "vd": "ví dụ",
            "qa": "hỏi đáp",
            "faq": "câu hỏi thường gặp",
            "qna": "hỏi đáp",
        }
        self._compile_abbr_regex()

    def _compile_abbr_regex(self) -> None:
        """Build regex chỉ 1 lần để tăng tốc expand."""
        escaped = map(re.escape, self._abbr.keys())
        self._re_abbr = re.compile(r"\b(" + "|".join(escaped) + r")\b", flags=re.I)

    # ------------------------------------------------------------------
    # 2. Mở rộng viết tắt
    # ------------------------------------------------------------------
    def expand(self, text: str) -> str:
        """Thay thế toàn bộ viết tắt trong text."""
        if not self._re_abbr:
            self._compile_abbr_regex()
        return self._re_abbr.sub(lambda m: self._abbr[m.group(0).lower()], text)

    # ------------------------------------------------------------------
    # 3. Tokenize
    # ------------------------------------------------------------------
    def tokenize(self, text: str) -> List[str]:
        """underthesea.word_tokenize + pyvi.ViTokenizer."""
        tokens = uts_word_tokenize(text)
        text = " ".join(tokens)
        tokens = ViTokenizer.tokenize(text).split()
        return tokens

    # ------------------------------------------------------------------
    # 4. Stop-words
    # ------------------------------------------------------------------
    def load_stopwords(self, path: str | Path) -> "AbbreviationProcessor":
        """Nạp stop-word từ file (một từ / dòng)."""
        path = Path(path)
        if path.exists():
            with path.open(encoding="utf-8") as f:
                self._stopwords = [w.strip() for w in f if w.strip()]
            # Ưu tiên cụm dài trước
            self._stopwords.sort(key=lambda w: len(w.split()), reverse=True)
        else:
            self._stopwords = []
        return self

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Loại bỏ stop-word khỏi danh sách token."""
        if not self._stopwords:
            return tokens
        text = " ".join(tokens)
        for sw in self._stopwords:
            pattern = r"\b" + re.escape(sw) + r"\b"
            text = re.sub(pattern, "", text, flags=re.I)
        # Dọn dẹp khoảng trắng thừa
        text = re.sub(r"\s+", " ", text).strip()
        return text.split() if text else []

    # ------------------------------------------------------------------
    # 5. Chunk / cụm từ
    # ------------------------------------------------------------------
    def extract_phrases(self, tokens: List[str]) -> List[str]:
        """POS-tag → gom cụm N/V/A/R/C."""
        if not tokens:
            return []
        tagged = pos_tag(" ".join(tokens))
        phrases, cur = [], []
        for w, p in tagged:
            if p in {"N", "Np", "Ny", "V", "A", "R", "C"}:
                cur.append(w)
            else:
                if cur:
                    phrases.append(" ".join(cur))
                    cur = []
        if cur:
            phrases.append(" ".join(cur))
        return phrases

    # ------------------------------------------------------------------
    # 6. Pipeline cuối
    # ------------------------------------------------------------------
    def run(self, text: str) -> str:
        """Chạy toàn bộ pipeline và trả về cụm đầu tiên."""
        text = text_normalize(text)
        text = self.expand(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        phrases = self.extract_phrases(tokens)
        return phrases[0] if phrases else ""


# ----------------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    proc = AbbreviationProcessor(
        abbr_path="viet_tat.json",
        stopwords_path="stopwords.txt",
    )
    raw = "Oke lắm nheee mà món cơm chiên ăn rất ngon nhưng đợi hơi lâu"
    print("INPUT :", raw)
    print("OUTPUT:", proc.run(raw))