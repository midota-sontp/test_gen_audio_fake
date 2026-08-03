# VIVOS → Fish Speech S2 fake-speech dataset generator

Sinh dataset **phát hiện deepfake giọng nói tiếng Việt**: mỗi audio thật trong VIVOS được
ghép một audio **giả** do **Fish Speech S2** clone lại — **cùng transcript, cùng speaker, giọng
được nhân bản** nhưng là audio do model sinh ra. Kết quả là các cặp Real/Fake.

Vì mỗi speaker xuất hiện ở cả hai lớp (speaker-matched) và real/fake nói **cùng nội dung**
(content-matched), model detector không thể "đi tắt" theo danh tính giọng hay nội dung.

## Cấu trúc đầu ra

```
dataset/
├── real/
│   └── VIVOSSPK01/ VIVOSSPK01_R001.wav ...
├── fake/
│   └── fishspeech/
│       └── VIVOSSPK01/ VIVOSSPK01_R001_fake.wav ...
├── reference/
│   └── VIVOSSPK01.wav          # giọng mẫu 10-20s để clone
└── metadata/
    └── metadata.csv
```

`metadata.csv`:

| audio_path | label | speaker | text | generator | split |
|---|---|---|---|---|---|
| real/VIVOSSPK01/VIVOSSPK01_R001.wav | 0 | VIVOSSPK01 | Xin chào | real | train |
| fake/fishspeech/VIVOSSPK01/VIVOSSPK01_R001_fake.wav | 1 | VIVOSSPK01 | Xin chào | FishSpeechS2 | train |

Mọi audio được chuẩn hoá **16 kHz / mono / PCM-16**. File hỏng bị bỏ qua và ghi log; fake chỉ
được ghi vào metadata sau khi xác nhận file tồn tại & đọc được.

## Cài đặt

```bash
pip install -r requirements.txt
bash setup_fish.sh          # clone fish-speech + tải weights fishaudio/s2-pro (public)
```

`setup_fish.sh` cài `fish_speech` và tải checkpoint vào `third_party/fish-speech/checkpoints/s2-pro`.

## Dữ liệu VIVOS

Tải từ Kaggle: <https://www.kaggle.com/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr>
rồi giải nén sao cho có `vivos/train/{prompts.txt,waves/}` và `vivos/test/...`.

## Chạy

```bash
python cli.py --dataset vivos --output dataset --config config.yaml
```

- `--dataset` đè `dataset_root`, `--output` đè `output_root`, `--splits train` để chỉ chạy 1 split.
- **Resume**: chạy lại sẽ **bỏ qua** audio đã sinh (kiểm tra file hợp lệ + đã có trong metadata).
  Dùng `--overwrite` để sinh lại từ đầu.
- Thanh tiến độ (tqdm) cho pha real và pha fake; log ở `dataset/logs/generate.log`.

> **Hiệu năng**: S2 (4B) trên MPS ~25–70s/câu → sinh full VIVOS (~11.6k câu) mất nhiều giờ; nên
> chạy overnight. Sinh fake được **tuần tự hoá trên accelerator** (chạy nhiều tiến trình 4B song
> song sẽ OOM); `num_workers` chỉ tăng tốc chuẩn hoá audio thật + dựng reference (I/O).

## Chạy HẾT trong Docker (một lệnh)

Image dựng trên **image CPU chính thức của Fish Audio** (`fishaudio/fish-speech:server-cpu-v2.0.0-beta`,
đã có sẵn `fish_speech` + torch-cpu trong venv `/app/.venv`), chỉ thêm 3 gói nhẹ — nên build nhanh,
không dính dependency hell. Container **tự bootstrap** ([docker/bootstrap.sh](docker/bootstrap.sh)):
tự tải weights S2 + tự tải VIVOS từ Kaggle (nếu chưa có) rồi sinh dataset. Không cần setup native gì cả.

```bash
docker compose run --rm generate
```

Lần đầu sẽ: tải weights `fishaudio/s2-pro` → tải VIVOS → sinh. Mọi thứ **persist trên host** (weights,
VIVOS, output) nên chạy lại là **resume**. Kết quả ở `./dataset`.

Các volume trong [docker-compose.yml](docker-compose.yml):

| host | container | vai trò |
|---|---|---|
| `./vivos` | `/data/vivos` | VIVOS — **tự tải vào đây** nếu trống (persist host) |
| `./dataset` | `/data/output` | output ghi ngược ra host |
| `./third_party/fish-speech/checkpoints` | `/work/.../checkpoints` | weights S2 (tải 1 lần, giữ lại) |
| `./config.yaml` | `/work/config.yaml` (ro) | sửa config không cần build lại |
| `~/.kaggle` | `/home/fish/.kaggle` (ro) | token Kaggle để tải VIVOS |

Tải VIVOS cần token Kaggle: mount sẵn `~/.kaggle`, hoặc đặt `KAGGLE_USERNAME`/`KAGGLE_KEY`. Nếu bạn
đã có VIVOS sẵn, cứ đặt vào `./vivos/train/...` — container sẽ bỏ qua bước tải.

> ⚠️ **Model S2 là 4B — cần nhiều RAM/VRAM.** Đây là điểm dễ hỏng nhất.
>
> **Lỗi thường gặp: `t2s step failed (exit -9)`** = SIGKILL = **hết RAM (OOM)**. Load 4B trên CPU fp32
> cần **~16GB**. Cách xử lý:
> - **CPU**: tăng RAM cho Docker (Desktop → Settings → Resources → Memory ≥ 16GB). `half: true` (fp16, ~8GB)
>   có thể giúp nhưng trên CPU đôi khi lỗi `not implemented for 'Half'` → nếu vậy quay lại tăng RAM.
> - **GPU (Linux + NVIDIA, khuyến nghị)**: sửa dòng `FROM` trong [Dockerfile](Dockerfile) sang
>   `fishaudio/fish-speech:server-cuda-v2.0.0-beta`, đặt `half: true`, rồi:
>   ```bash
>   FISH_DEVICE=cuda docker compose run --rm generate   # + bỏ comment khối deploy: GPU trong compose
>   ```
>
> **Chậm**: wrapper gọi CLI theo từng câu nên model bị **nạp lại mỗi clip** (~40-60s/clip chỉ để load).
> Trên CPU nên để `limit` nhỏ; sinh số lượng lớn nên dùng GPU.
>
> Bật fp16 trong `config.yaml`:
> ```yaml
> fishspeech:
>   half: true
> ```

## Đồng bộ lên Kaggle (chạy nhiều phiên, không mất tiến độ)

Chạy full VIVOS (~12.4k câu) mất khoảng **30 giờ GPU** → vượt giới hạn 1 phiên Kaggle, và
`/kaggle/working` **bị xoá khi phiên kết thúc**. Bật `kaggle_sync` để dataset sống trên
Kaggle thay vì trong thư mục tạm:

```yaml
kaggle_sync:
  enabled: true
  handle: "sonpham12/vivos-fake"
  single_dataset: true
  every_clips: 1
```

hoặc `python cli.py --kaggle-handle <user>/<slug> ...`; chạy local không có token thì `--no-kaggle-sync`.

- **Khởi động**: tải toàn bộ shard đã có về `output_root` (từng file một), gộp
  `metadata.csv` → chạy tiếp đúng chỗ đã dừng (kết hợp với logic resume sẵn có). File đã có
  ở local thì **giữ nguyên**, không bị ghi đè.
- **Trong lúc chạy**: cứ mỗi `every_clips` fake mới (hoặc `every_minutes` phút) thì
  checkpoint. Upload chạy **thread nền** nên không chặn việc sinh audio; nếu lần upload
  trước chưa xong thì bỏ qua nhịp này.
- **Kết thúc / Ctrl-C / crash**: `finally` luôn đẩy nốt phần còn lại.

Audio được gói vào một **payload zip** (`.sync/vivos-fake-data.zip`) rồi upload cùng
`metadata.csv`. Hai chế độ:

- `single_dataset: true` — tất cả nằm trong đúng một dataset. Mỗi lần đẩy là **upload lại
  toàn bộ payload** (Kaggle Dataset là *artifact có version*, không phải filesystem —
  kagglehub không có delta), nên payload càng lớn thì mỗi lần đẩy càng lâu: vài giây lúc
  đầu, ~10 phút khi dataset đã 3.5GB.
- `single_dataset: false` — chia **shard**: `<handle>-001`, `-002`, ... Checkpoint chỉ
  upload lại shard đang mở; shard vượt `shard_mb` thì niêm phong (không đụng lại, xoá zip
  local cho nhẹ đĩa). Tổng upload cả run ≈ kích thước dataset thay vì O(n²).

Với `every_clips: 1` thì mỗi audio hợp lệ đều kích hoạt đẩy; nhịp nào rơi vào lúc đang
upload thì bị gộp, nên thực tế là "đẩy liên tục hết mức có thể" chứ không xếp hàng chồng lên nhau.

### Hai hành vi của Kaggle quyết định cách resume (⚠️ sai là mất dữ liệu)

1. **Kaggle tự giải nén file zip mình upload.** `vivos-fake-data.zip` lên tới Kaggle thì
   thành các file rời trong thư mục `vivos-fake-data/...` — **không tải lại được cái zip**.
   Nên zip chỉ là *bao bì để upload*, còn `pull()` phục hồi **từng file rời** (tự bỏ tiền tố
   `vivos-fake-data/`, `shard-001/`, hay `dataset/` của luồng cũ).
2. **Mỗi version publish payload NGUYÊN KHỐI** — thứ gì không có trong payload là Kaggle
   **xoá** khỏi dataset. Nên payload luôn phải mang theo mọi file đã publish ở shard đang
   mở: `.sync/state.json` ghi lại danh sách đó (`open_files`), và trước mỗi lần đẩy,
   `push()` **gói lại** những file đã publish mà payload local đang thiếu (máy mới,
   `/kaggle/working` bị xoá, xoá `.sync/`).

Nếu một file đã publish mà **vừa** không có trong payload **vừa** không có trên đĩa thì
lần đẩy đó bị **từ chối** (log `refusing to publish: ...`) thay vì publish một version xoá
mất nó. Muốn cho dataset co lại thật thì đặt `allow_delete: true`.

| khoá | mặc định | ý nghĩa |
|---|---|---|
| `enabled` | `false` | bật/tắt |
| `handle` | — | `<username>/<slug>` |
| `single_dataset` | `false` | `true` = **tất cả** nằm trong đúng `handle` (không chia shard, không niêm phong) |
| `pull_on_start` | `true` | tải dữ liệu đã có về trước khi chạy |
| `every_clips` | `200` | checkpoint sau mỗi N fake mới (`1` = sau mỗi audio; 0 = tắt) |
| `every_minutes` | `20` | ...hoặc sau ngần này phút (0 = tắt) |
| `shard_mb` | `400` | ngưỡng niêm phong shard — bỏ qua khi `single_dataset: true` |
| `allow_delete` | `false` | `true` = cho phép publish version thiếu file đã publish (dataset co lại) |
| `include` | `[real, fake, reference]` | **đẩy những gì lên dataset** (xem dưới) |
| `exclude_patterns` | `[]` | lọc thêm theo wildcard, vd `["real/VIVOSSPK01/*"]` |
| `include_real` | — | cách viết tắt cũ; `false` = bỏ `real` khỏi `include` |

### Đẩy những gì lên dataset

| giá trị trong `include` | nội dung | bỏ đi thì sao |
|---|---|---|
| `real` | `real/` — audio thật đã chuẩn hoá | phiên sau tự sinh lại từ VIVOS (~1 phút), nhưng dataset trên Kaggle sẽ thiếu lớp real |
| `fake` | `fake/fishspeech/` — audio giả | mất thứ đắt nhất; đừng bỏ |
| `reference` | `reference/` — giọng mẫu 10–20s mỗi speaker | rất nhẹ, phải dựng lại từ VIVOS |
| `logs` | `logs/*.log` của phiên hiện tại | không ảnh hưởng dữ liệu |

`metadata.csv` **luôn** được đẩy — nó chính là trạng thái resume, bỏ đi là mất khả năng chạy
tiếp. Nó và `logs/` nằm **rời** cạnh zip và được làm mới mỗi lần đẩy (vì chúng thay đổi liên
tục); audio thì gói vào zip một lần cho mỗi shard.

Chỉ file **đã có trong `metadata.csv`** mới được upload — mà một dòng metadata chỉ được ghi
sau khi audio đã validate, nên wav ghi dở không bao giờ lên Kaggle. Dataset tạo ra ở chế độ
**private**. Sai tên trong `include` thì dừng ngay lúc khởi động kèm danh sách hợp lệ.

Đổi `include` giữa chừng chỉ ảnh hưởng file **thêm mới từ đó trở đi** — phần đã nằm trong
payload đã publish thì vẫn còn; muốn dựng lại sạch thì xoá `<output>/.sync/` rồi đẩy lại
(sẽ có cảnh báo trong log khi phát hiện `include` đổi).

**Auth** — lấy token ở <https://www.kaggle.com/settings/api> ("Generate New Token", hoặc
"Create Legacy API Key" nếu muốn cặp username/key). kagglehub dò theo đúng thứ tự này:

1. token ngầm của phiên Kaggle notebook (biến `KAGGLE_API_V1_TOKEN_PATH`)
2. `KAGGLE_API_TOKEN`
3. `~/.kaggle/access_token`
4. `KAGGLE_USERNAME` + `KAGGLE_KEY`
5. `~/.kaggle/kaggle.json`

⚠️ Trong Kaggle notebook, (1) **thắng** mọi credential bạn tự đặt. Nếu token phiên không đủ
quyền ghi dataset, phải `os.environ.pop("KAGGLE_API_V1_TOKEN_PATH", None)` trước rồi mới đặt
token của mình.

Token được kiểm tra ngay lúc khởi động (`whoami`) để sai token thì hỏng trong vài giây, không
phải sau nhiều giờ sinh audio.

## Cấu hình (`config.yaml`)

| khoá | mặc định | ý nghĩa |
|---|---|---|
| `dataset_root` | `vivos` | thư mục chứa `train/`, `test/` |
| `output_root` | `dataset` | nơi ghi `real/ fake/ reference/ metadata/` |
| `splits` | `[train, test]` | các split VIVOS xử lý |
| `limit` | `null` | **số audio cần gen**: `null`=tất cả, hoặc số nguyên (vd `200` → 200 real + 200 fake) |
| `max_per_speaker` | `null` | giới hạn số câu mỗi speaker (`null`=không giới hạn) |
| `reference_seconds` | `15` | độ dài reference/speaker (giới hạn 10–20s) |
| `sample_rate` | `16000` | tần số lấy mẫu đầu ra |
| `generator` | `FishSpeechS2` | backend sinh giả (factory trong `src/fishspeech.py`) |
| `num_workers` | `4` | song song hoá pha real + reference |
| `overwrite` | `false` | `false` = resume (bỏ qua file đã có) |
| `kaggle_sync.*` | tắt | checkpoint dataset lên Kaggle + kéo về khi khởi động (xem mục trên) |
| `fishspeech.*` | | device/temperature/top_p/top_k/seed/step_timeout của S2 |

## Kiến trúc mã (module hoá)

| module | vai trò |
|---|---|
| `src/parser.py` | quét corpus, đọc `prompts.txt` → `Utterance(audio_id, speaker, text, wav, split)`, gom theo speaker |
| `src/preprocess.py` | load/chuẩn hoá 16k·mono·PCM-16, validate, bỏ file hỏng |
| `src/reference_builder.py` | dựng giọng mẫu 10–20s/speaker (ghép clip dài nhất) |
| `src/fishspeech.py` | backend Fish Speech S2 (CLI 3 bước: encode → text2semantic → decode), cache prompt tokens |
| `src/metadata.py` | ghi `metadata.csv` tăng dần, resume-safe, thread-safe |
| `src/kaggle_sync.py` | checkpoint theo shard lên Kaggle (thread nền) + pull/gộp metadata khi khởi động |
| `src/generator.py` | điều phối: pha real+reference (song song) → pha fake (tuần tự) |
| `cli.py` | argparse + đọc YAML + chạy |

Muốn thêm backend khác: cài đặt lớp `Generator` (2 hàm `prepare_speaker`, `generate`) và đăng ký
trong `get_generator` (`src/fishspeech.py`) — phần còn lại không đổi.
