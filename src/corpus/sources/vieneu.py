"""VieNeu-TTS-140h (pnnbao-ump/VieNeu-TTS-140h) — nguồn FAKE.

Dataset *gated*: phải accept terms trên HuggingFace rồi truyền HF_TOKEN, nếu không
mọi request trả 401. Repo lưu ở dạng `save_to_disk`: 49 file `.arrow` (~485 MB mỗi
file, tổng ~24 GB — audio là PCM thô chứ không nén).

Vì không đọc được schema khi chưa có quyền, tên cột được **dò lúc chạy** từ schema
Arrow; có thể ép bằng tham số nếu dò sai (xem `scripts/build_fake.py --probe`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pyarrow as pa

from .base import RawItem, download, hf_url

REPO = "pnnbao-ump/VieNeu-TTS-140h"
N_FILES = 49
EXPECTED_ROWS = 74858

SPEAKER_NAMES = ("speaker_id", "speaker", "speaker_name", "spk_id", "spk",
                 "voice", "voice_id", "speaker_idx")
TEXT_NAMES = ("text", "transcript", "transcription", "sentence", "normalized_text",
              "raw_text", "phonemes", "phoneme")
ID_NAMES = ("id", "utterance_id", "utt_id", "audio_id", "file_id", "name")


class SchemaError(RuntimeError):
    pass


def _detect(schema: pa.Schema) -> dict[str, str | None]:
    """Đoán vai trò từng cột từ schema Arrow."""
    names = list(schema.names)
    lower = {n.lower(): n for n in names}

    audio = None
    for n in names:
        t = schema.field(n).type
        if pa.types.is_struct(t) and {f.name for f in t} & {"bytes", "array", "path"}:
            audio = n
            break
    if audio is None:
        for cand in ("audio", "wav", "speech"):
            if cand in lower:
                audio = lower[cand]
                break
    if audio is None:
        raise SchemaError(f"Không tìm thấy cột audio trong schema: {names}")

    def pick(cands):
        for c in cands:
            if c in lower:
                return lower[c]
        return None

    return {"audio": audio, "speaker": pick(SPEAKER_NAMES),
            "text": pick(TEXT_NAMES), "id": pick(ID_NAMES)}


def _tag(value) -> str:
    """Thêm tiền tố `vieneu_` nhưng không lặp nếu giá trị gốc đã có sẵn."""
    v = str(value)
    return v if v.lower().startswith("vieneu") else f"vieneu_{v}"


def _open(path: Path):
    src = pa.memory_map(str(path), "rb")
    try:
        return pa.ipc.open_stream(src)          # save_to_disk dùng IPC stream
    except pa.ArrowInvalid:
        src.seek(0)
        return pa.ipc.open_file(src)


class VieNeuSource:
    name = "vieneu_tts"
    label = 1
    generator = "vieneu_tts"

    def __init__(self, cache_dir: Path, hf_token: str | None = None,
                 keep_cache: bool = True, columns: dict | None = None):
        self.cache = Path(cache_dir) / self.name
        self.token = hf_token
        self.keep_cache = keep_cache
        self.columns = columns or {}
        if not self.token:
            raise SchemaError(
                "VieNeu-TTS-140h là dataset gated: accept terms tại "
                f"https://huggingface.co/datasets/{REPO} rồi đặt HF_TOKEN.")

    # ---------- hạ tầng ----------
    def units(self) -> list[str]:
        return [f"data-{i:05d}-of-{N_FILES:05d}.arrow" for i in range(N_FILES)]

    def expected_rows(self) -> int | None:
        return EXPECTED_ROWS

    def prepare(self) -> None:
        """Không tải sẵn: 24 GB. Mỗi unit tải đúng lúc cần trong iter_unit()."""
        self.cache.mkdir(parents=True, exist_ok=True)

    def _fetch(self, unit: str) -> Path:
        return download(hf_url(REPO, unit), self.cache / unit, self.token,
                        desc=f"vieneu/{unit}")

    def probe(self, unit: str | None = None) -> dict:
        """Tải 1 file rồi in schema — dùng để xác nhận việc dò cột."""
        path = self._fetch(unit or self.units()[0])
        reader = _open(path)
        schema = reader.schema
        detected = _detect(schema)
        detected.update({k: v for k, v in self.columns.items() if v})
        return {"fields": [(f.name, str(f.type)[:80]) for f in schema],
                "detected": detected}

    # ---------- duyệt ----------
    def iter_unit(self, unit: str, start_row: int) -> Iterator[tuple[int, RawItem]]:
        path = self._fetch(unit)
        try:
            reader = _open(path)
            cols = _detect(reader.schema)
            cols.update({k: v for k, v in self.columns.items() if v})
            file_tag = unit.split("-")[1]
            row_idx = 0
            for batch in reader:
                n = batch.num_rows
                if row_idx + n <= start_row:
                    row_idx += n
                    continue
                data = {role: batch.column(name).to_pylist()
                        for role, name in cols.items() if name}
                for i in range(n):
                    if row_idx + i < start_row:
                        continue
                    yield row_idx + i, self._to_item(data, i, row_idx + i,
                                                     file_tag, cols)
                row_idx += n
        finally:
            if not self.keep_cache and path.exists():
                path.unlink()       # 24 GB không phải máy nào cũng chứa nổi

    def _to_item(self, data: dict, i: int, row: int, file_tag: str,
                 cols: dict) -> RawItem:
        audio = data["audio"][i]
        if isinstance(audio, dict):
            blob = audio.get("bytes")
            if blob is None:
                raise SchemaError(
                    f"Cột audio '{cols['audio']}' không chứa bytes: {list(audio)}")
        else:
            raise SchemaError(f"Cột audio '{cols['audio']}' kiểu lạ: {type(audio)}")

        raw_id = data.get("id", [None] * (i + 1))[i] if cols.get("id") else None
        item_id = _tag(raw_id) if raw_id else f"vieneu_{file_tag}_{row:06d}"
        spk = data.get("speaker", [None] * (i + 1))[i] if cols.get("speaker") else None
        speaker_id = _tag(spk) if spk is not None else f"vieneu_unknown_{file_tag}"
        return RawItem(
            item_id=item_id,
            source=self.name,
            speaker_id=speaker_id,
            speaker_known=spk is not None,
            recording_id=item_id,          # mỗi câu TTS là một bản ghi độc lập
            original_split="train",
            audio_bytes=blob,
            label=self.label,
            generator=self.generator,
            extra={"text": data.get("text", [None] * (i + 1))[i] if cols.get("text") else None,
                   "arrow_file": file_tag},
        )
