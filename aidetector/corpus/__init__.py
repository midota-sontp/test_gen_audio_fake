"""CHUẨN CORPUS — định dạng dữ liệu thống nhất của dự án.

Mọi nguồn dữ liệu (VIVOS, Common Voice, thư mục wav bất kỳ, HuggingFace, audio do
TTS/voice-cloning sinh ra) đều được ép về đúng một dạng duy nhất mô tả trong
`spec.py`, rồi ghi vào `corpus/manifest.csv` theo schema trong `schema.py`.

Nhờ vậy các tầng phía sau (augment → backbone → classifier) không cần biết dữ liệu
đến từ đâu.
"""

from .schema import COLUMNS, LABEL_FAKE, LABEL_REAL, Record  # noqa: F401
from .spec import AudioSpec, DEFAULT_SPEC, QualityIssue, check_quality  # noqa: F401
from .manifest import Manifest  # noqa: F401
