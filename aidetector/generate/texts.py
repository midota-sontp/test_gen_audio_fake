"""Nguồn văn bản cho TTS.

Ưu tiên dùng transcript của chính corpus REAL: khi câu nói của fake trùng nội dung
với real, mô hình không thể "ăn gian" bằng cách học chủ đề/ngữ nghĩa mà buộc phải
nhìn vào dấu vết tổng hợp.

Danh sách dự phòng bên dưới chỉ dùng khi corpus không có transcript nào.
"""

from __future__ import annotations

from pathlib import Path

FALLBACK_SENTENCES = [
    "Hôm nay trời rất đẹp, thích hợp cho một chuyến đi chơi xa cùng gia đình.",
    "Công nghệ trí tuệ nhân tạo đang thay đổi cách chúng ta làm việc mỗi ngày.",
    "Xin vui lòng chờ trong giây lát, nhân viên sẽ hỗ trợ quý khách ngay bây giờ.",
    "Giá xăng dầu trong nước tiếp tục được điều chỉnh vào chiều nay theo chu kỳ.",
    "Anh ấy đã dành cả buổi sáng để chuẩn bị tài liệu cho cuộc họp quan trọng.",
    "Học sinh cần rèn luyện thói quen đọc sách để mở rộng vốn từ và tư duy.",
    "Ngân hàng khuyến cáo khách hàng tuyệt đối không cung cấp mã OTP cho người lạ.",
    "Đội tuyển quốc gia sẽ có trận đấu quyết định vào tối thứ bảy tuần này.",
    "Thời tiết miền Bắc chuyển lạnh, nhiệt độ thấp nhất có thể xuống dưới mười độ.",
    "Chúng tôi cam kết bảo mật thông tin cá nhân của người dùng theo quy định.",
    "Bạn hãy nhấn phím một để nghe lại nội dung, hoặc phím hai để gặp tổng đài viên.",
    "Con đường làng quanh co dẫn ra cánh đồng lúa đang vào mùa gặt rộ.",
    "Việc tập thể dục đều đặn mỗi ngày giúp cải thiện sức khỏe tim mạch rõ rệt.",
    "Buổi hội thảo sẽ diễn ra tại hội trường tầng ba, bắt đầu lúc chín giờ sáng.",
    "Món phở bò truyền thống Hà Nội có nước dùng trong và thơm mùi quế hồi.",
    "Hệ thống sẽ tạm ngưng để bảo trì từ nửa đêm đến bốn giờ sáng mai.",
    "Tôi nghĩ rằng chúng ta nên xem xét lại phương án này một cách cẩn thận hơn.",
    "Các bạn nhỏ vui vẻ chạy nhảy trong sân trường sau tiếng trống tan học.",
    "Chuyến tàu khởi hành từ ga Sài Gòn sẽ đến ga Hà Nội vào sáng ngày hôm sau.",
    "Nghiên cứu mới cho thấy giấc ngủ đủ giấc ảnh hưởng lớn đến khả năng ghi nhớ.",
]


def load_texts(path: str | Path) -> list[str]:
    """Đọc file văn bản, mỗi dòng một câu."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def is_usable(text: str, min_words: int = 6, max_words: int = 40) -> bool:
    """Lọc câu quá ngắn/quá dài để audio sinh ra rơi vào khoảng 3–10 giây."""
    n = len(text.split())
    return min_words <= n <= max_words
