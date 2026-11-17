import hashlib
import io
import json
import os
import time
import zipfile
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

import torch
from PIL import Image, UnidentifiedImageError

from config import dispositivo
from dataset import transform_avaliacao
from modelo import carregar_checkpoint


ARCHIVE_URL = (
    "https://data.mendeley.com/public-files/datasets/2svdj3yyrk/files/"
    "8675115d-b26c-407f-b405-7cc777c1f922/file_downloaded"
)
ARCHIVE_SIZE = 1_264_600_271
ARCHIVE_SHA256 = "ed45f1f3ec3cf2fa083b026ddce6d8666d9575d204f19ca6b92ebad3e4f5ffc7"
DATA_ROOT = Path("data_public")
MODEL_PATH = Path(
    os.environ.get(
        "PLANT_EVAL_MODEL",
        str(Path("modelos_multiclasse") / "melhor_modelo.pth"),
    )
)
OUTPUT_PATH = Path(
    os.environ.get(
        "PLANT_EVAL_OUTPUT",
        str(
            Path("resultados_validacao_externa")
            / "mendeley_bitter_gourd_baseline.json"
        ),
    )
)
SELECTED_CROP = "Melão-amargo"
CONFIDENCE_THRESHOLD = 0.70
BATCH_SIZE = 64
BLOCK_SIZE = 2 * 1024 * 1024
CACHE_BLOCKS = 12


class RangeFile(io.RawIOBase):
    def __init__(self) -> None:
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.bytes_requested = 0
        self.request_count = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.position = offset
        elif whence == 1:
            self.position += offset
        elif whence == 2:
            self.position = ARCHIVE_SIZE + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        return self.position

    def _read_block(self, start: int) -> bytes:
        cached = self.cache.get(start)
        if cached is not None:
            self.cache.move_to_end(start)
            return cached

        end = min(ARCHIVE_SIZE - 1, start + BLOCK_SIZE - 1)
        request = Request(
            ARCHIVE_URL,
            headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": "Mozilla/5.0",
            },
        )
        with urlopen(request, timeout=120) as response:
            if response.status != 206:
                raise OSError(f"Archive range request returned HTTP {response.status}")
            data = response.read()
        if len(data) != end - start + 1:
            raise OSError(f"Incomplete archive range at byte {start}")

        self.request_count += 1
        self.bytes_requested += len(data)
        self.cache[start] = data
        if len(self.cache) > CACHE_BLOCKS:
            self.cache.popitem(last=False)
        return data

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = ARCHIVE_SIZE - self.position
        size = min(size, max(0, ARCHIVE_SIZE - self.position))
        if size == 0:
            return b""

        end = self.position + size
        cursor = self.position
        chunks = []
        while cursor < end:
            block_start = cursor // BLOCK_SIZE * BLOCK_SIZE
            block = self._read_block(block_start)
            offset = cursor - block_start
            count = min(end - cursor, len(block) - offset)
            if count <= 0:
                break
            chunks.append(block[offset : offset + count])
            cursor += count
        self.position = cursor
        return b"".join(chunks)


def image_hashes(root: Path) -> set[str]:
    hashes = set()
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for split in ("train", "val", "test"):
        folder = root / split
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                digest = hashlib.sha256()
                with path.open("rb") as image_file:
                    for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                        digest.update(chunk)
                hashes.add(digest.hexdigest())
    return hashes


def main() -> None:
    print("Loading model and class metadata...", flush=True)
    device = dispositivo()
    model, classes = carregar_checkpoint(MODEL_PATH, device)
    class_metadata = json.loads(
        (DATA_ROOT / "classes.json").read_text(encoding="utf-8")
    )
    transform = transform_avaliacao()

    indices_by_crop: dict[str, list[int]] = defaultdict(list)
    for index, class_name in enumerate(classes):
        crop = class_metadata.get(class_name, {}).get("planta")
        if crop:
            indices_by_crop[crop].append(index)
    if SELECTED_CROP not in indices_by_crop:
        raise ValueError(f"{SELECTED_CROP} is not present in the model metadata")
    selected_indices = indices_by_crop[SELECTED_CROP]

    print("Hashing existing train/validation/test images...", flush=True)
    existing_hashes = image_hashes(DATA_ROOT)
    print(f"Found {len(existing_hashes)} distinct existing image hashes.", flush=True)

    remote = RangeFile()
    records: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    duplicate_training_count = 0
    duplicate_source_count = 0
    corrupt_images: list[dict[str, str]] = []
    pending_images: list[torch.Tensor] = []
    pending_metadata: list[dict[str, str]] = []
    source_counts: Counter[str] = Counter()

    def evaluate_batch() -> None:
        if not pending_images:
            return
        inputs = torch.stack(pending_images).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(inputs), dim=1).cpu()

        for row, source in zip(probabilities, pending_metadata):
            top1_index = int(row.argmax().item())
            crop_scores = {
                crop: float(row[indices].sum().item())
                for crop, indices in indices_by_crop.items()
            }
            predicted_crop = max(crop_scores, key=crop_scores.get)
            selected_index = max(
                selected_indices, key=lambda index: float(row[index].item())
            )
            selected_confidence = float(row[selected_index].item())
            crop_mismatch = predicted_crop != SELECTED_CROP
            uncertain = crop_mismatch or selected_confidence < CONFIDENCE_THRESHOLD
            api_status = (
                None
                if uncertain
                else (
                    "healthy"
                    if class_metadata[classes[selected_index]]["saudavel"]
                    else "diseased"
                )
            )
            true_label = source["true_label"]
            records.append(
                {
                    "file": source["file"],
                    "true_label": true_label,
                    "true_status": "healthy" if true_label == "Healthy" else "diseased",
                    "top1_class": classes[top1_index],
                    "top1_crop": class_metadata.get(classes[top1_index], {}).get(
                        "planta"
                    ),
                    "predicted_crop_by_aggregate": predicted_crop,
                    "selected_crop_class": classes[selected_index],
                    "selected_crop_confidence": selected_confidence,
                    "crop_mismatch": crop_mismatch,
                    "api_uncertain": uncertain,
                    "api_status": api_status,
                    "global_top1_confidence": float(row[top1_index].item()),
                    "top3": [
                        {
                            "class": classes[int(index)],
                            "score": float(row[int(index)].item()),
                        }
                        for index in row.topk(3).indices
                    ],
                }
            )
        pending_images.clear()
        pending_metadata.clear()

    print("Reading original bitter-gourd images from the public archive...", flush=True)
    prefix = "Dataset/Raw Data/Bitter gourd/"
    with zipfile.ZipFile(remote) as archive:
        source_infos = [
            info
            for info in archive.infolist()
            if not info.is_dir()
            and info.filename.startswith(prefix)
            and info.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        for info in source_infos:
            true_label = info.filename[len(prefix) :].split("/")[0]
            source_counts[true_label] += 1
        expected_classes = {"Anthracnose", "Downey Mildew", "Healthy"}
        if set(source_counts) != expected_classes:
            raise ValueError(f"Unexpected raw dataset classes: {dict(source_counts)}")

        start_time = time.monotonic()
        for number, info in enumerate(source_infos, 1):
            true_label = info.filename[len(prefix) :].split("/")[0]
            raw_image = archive.read(info)
            digest = hashlib.sha256(raw_image).hexdigest()
            if digest in existing_hashes:
                duplicate_training_count += 1
                continue
            if digest in seen_hashes:
                duplicate_source_count += 1
                continue
            seen_hashes.add(digest)
            try:
                with Image.open(io.BytesIO(raw_image)) as source_image:
                    pending_images.append(transform(source_image.convert("RGB")))
            except UnidentifiedImageError as error:
                corrupt_images.append(
                    {"file": info.filename, "error": str(error)}
                )
                continue
            pending_metadata.append(
                {"file": info.filename, "true_label": true_label}
            )
            if len(pending_images) >= BATCH_SIZE:
                evaluate_batch()
            if number % 200 == 0:
                print(
                    f"Processed {number}/{len(source_infos)}; "
                    f"predictions={len(records) + len(pending_images)}; "
                    f"archive bytes fetched={remote.bytes_requested}",
                    flush=True,
                )
        evaluate_batch()
        elapsed = time.monotonic() - start_time

    healthy_rows = [row for row in records if row["true_status"] == "healthy"]
    diseased_rows = [row for row in records if row["true_status"] == "diseased"]
    decisive_rows = [row for row in records if not row["api_uncertain"]]
    by_label = {}
    for label in sorted(source_counts):
        rows = [row for row in records if row["true_label"] == label]
        total = len(rows)
        decisive = [row for row in rows if not row["api_uncertain"]]
        by_label[label] = {
            "count_after_deduplication": total,
            "api_uncertain_count": sum(row["api_uncertain"] for row in rows),
            "api_coverage": len(decisive) / total if total else None,
            "api_disease_predictions": sum(
                row["api_status"] == "diseased" for row in rows
            ),
            "api_healthy_false_positive_rate": (
                sum(row["api_status"] == "diseased" for row in rows) / total
                if label == "Healthy" and total
                else None
            ),
            "api_disease_detection_rate": (
                sum(row["api_status"] == "diseased" for row in rows) / total
                if label != "Healthy" and total
                else None
            ),
            "top1_crop_accuracy": (
                sum(row["top1_crop"] == SELECTED_CROP for row in rows) / total
                if total
                else None
            ),
            "top1_downy_mildew_accuracy": (
                sum(
                    row["top1_class"] == "bitter_gourd__downey_mildew"
                    for row in rows
                )
                / total
                if label == "Downey Mildew" and total
                else None
            ),
            "top1_healthy_accuracy": (
                sum(
                    class_metadata[row["top1_class"]]["saudavel"] for row in rows
                )
                / total
                if label == "Healthy" and total
                else None
            ),
        }

    report = {
        "source": {
            "title": (
                "Leaf Image Dataset for Disease Detection in Bitter Gourd, "
                "Okra, Pumpkin, and Ridge Gourd"
            ),
            "doi": "10.17632/2svdj3yyrk.2",
            "url": "https://data.mendeley.com/datasets/2svdj3yyrk/2",
            "license": "CC BY 4.0",
            "attribution": (
                "Md Forhadul Isalm, Imon Sutradhar, and Md Mizanur Rahman, "
                "Mendeley Data (2025), DOI 10.17632/2svdj3yyrk.2"
            ),
            "archive_sha256": ARCHIVE_SHA256,
            "archive_size_bytes": ARCHIVE_SIZE,
            "only_original_raw_images_used": True,
            "raw_source_counts": dict(source_counts),
        },
        "evaluation": {
            "model_checkpoint": str(MODEL_PATH),
            "selected_crop": SELECTED_CROP,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "exact_training_duplicates_excluded": duplicate_training_count,
            "exact_source_duplicates_excluded": duplicate_source_count,
            "corrupt_images_skipped": corrupt_images,
            "unique_images_evaluated": len(records),
            "range_requests": remote.request_count,
            "archive_bytes_fetched": remote.bytes_requested,
            "elapsed_seconds": round(elapsed, 1),
            "api_decisive_coverage": (
                len(decisive_rows) / len(records) if records else None
            ),
            "api_healthy_false_positive_rate": (
                sum(row["api_status"] == "diseased" for row in healthy_rows)
                / len(healthy_rows)
                if healthy_rows
                else None
            ),
            "api_disease_detection_rate_including_uncertain": (
                sum(row["api_status"] == "diseased" for row in diseased_rows)
                / len(diseased_rows)
                if diseased_rows
                else None
            ),
            "api_health_accuracy_on_decisive": (
                sum(row["api_status"] == row["true_status"] for row in decisive_rows)
                / len(decisive_rows)
                if decisive_rows
                else None
            ),
            "global_top1_crop_accuracy": (
                sum(row["top1_crop"] == SELECTED_CROP for row in records)
                / len(records)
                if records
                else None
            ),
            "by_source_class": by_label,
        },
        "notable_health_errors": sorted(
            [
                row
                for row in records
                if (
                    row["true_status"] == "healthy"
                    and row["api_status"] == "diseased"
                )
                or (
                    row["true_status"] == "diseased"
                    and row["api_status"] == "healthy"
                )
            ],
            key=lambda row: row["selected_crop_confidence"],
            reverse=True,
        )[:40],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(report["evaluation"], ensure_ascii=False, indent=2),
        flush=True,
    )
    print(f"Report saved to {OUTPUT_PATH.resolve()}", flush=True)


if __name__ == "__main__":
    main()
