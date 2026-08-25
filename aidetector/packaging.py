"""Đóng gói / bung corpus thành một file zip.

Cần cho các môi trường phiên ngắn như Kaggle và Colab: dữ liệu trong thư mục làm
việc mất khi phiên kết thúc, và việc lưu hàng chục nghìn file wav rời rạc ra output
thì cực chậm. Gói tất cả vào một zip rồi đăng lên Kaggle Dataset là cách chuyển
corpus giữa các phiên nhanh nhất.

    Phiên 1:  ingest → generate → pack   → tải zip lên Kaggle Dataset
    Phiên 2:  unpack → split → features → train → evaluate
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from .corpus.manifest import LEGACY_MANIFEST_NAME, MANIFEST_NAME, Manifest, find_manifest
from .utils import ensure_dir, get_logger, progress

log = get_logger("aidetector.packaging")


def pack_corpus(
    corpus_root: str | Path,
    out_path: str | Path,
    include_audio: bool = True,
    compress: bool = False,
) -> Path:
    """Gói corpus thành một zip duy nhất.

    WAV PCM đã là dữ liệu khó nén; mặc định dùng `ZIP_STORED` để đóng gói nhanh
    hơn nhiều mà dung lượng gần như không đổi. Bật `compress` nếu cần tiết kiệm.
    """
    corpus_root = Path(corpus_root)
    manifest = Manifest.load(corpus_root, required=True)
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    missing = 0
    with zipfile.ZipFile(out_path, "w", mode, allowZip64=True) as zf:
        # Gói luôn dùng tên mới, kể cả khi corpus trên đĩa còn tên cũ.
        zf.write(find_manifest(corpus_root), MANIFEST_NAME)
        if include_audio:
            for rec in progress(list(manifest), total=len(manifest), label="pack"):
                src = corpus_root / rec.path
                if not src.exists():
                    missing += 1
                    continue
                zf.write(src, rec.path)

    size_mb = out_path.stat().st_size / 1024**2
    log.info("Đã gói %d bản ghi → %s (%.1f MB)", len(manifest), out_path, size_mb)
    if missing:
        log.warning("%d bản ghi thiếu file audio, không được đóng gói", missing)
    return out_path


def unpack_corpus(zip_path: str | Path, corpus_root: str | Path, overwrite: bool = False) -> Manifest:
    """Bung zip vào thư mục corpus và trả về manifest đã nạp."""
    zip_path = Path(zip_path)
    corpus_root = ensure_dir(corpus_root)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        # Chấp cả tên cũ: mọi corpus.zip đã đẩy lên Kaggle ở các phiên trước dùng nó,
        # và từ chối đọc chúng nghĩa là vứt bỏ hàng giờ GPU đã trả.
        if not ({MANIFEST_NAME, LEGACY_MANIFEST_NAME} & set(names)):
            raise ValueError(
                f"{zip_path} không phải gói corpus: thiếu {MANIFEST_NAME}"
            )
        todo = names if overwrite else [n for n in names if not (corpus_root / n).exists()]
        log.info("Bung %d / %d mục vào %s", len(todo), len(names), corpus_root)
        for name in progress(todo, total=len(todo), label="unpack"):
            zf.extract(name, corpus_root)

    manifest = Manifest.load(corpus_root, required=True)
    gone = manifest.prune_missing()
    if gone:
        manifest.save()
    log.info("Corpus sẵn sàng: %d bản ghi", len(manifest))
    return manifest
