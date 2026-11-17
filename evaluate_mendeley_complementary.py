import argparse
import hashlib
import io
import json
import os
import random
import re
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


BLOCK_SIZE = 2 * 1024 * 1024
CACHE_BLOCKS = 12
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
            / "mendeley_complementary_baseline.json"
        ),
    )
)

SOURCES = [
    {
        "id": "cucumber",
        "title": "Cucumber Disease Recognition Dataset",
        "doi": "10.17632/y6d3z6f8z9.1",
        "url": "https://data.mendeley.com/datasets/y6d3z6f8z9/1",
        "license": "CC BY 4.0",
        "attribution": (
            "Nusrat Sultana, Sumaita Binte Shorif, Morium Akter, and "
            "Mohammad Shorif Uddin, Mendeley Data, DOI 10.17632/y6d3z6f8z9.1"
        ),
        "archive_url": (
            "https://data.mendeley.com/public-files/datasets/y6d3z6f8z9/files/"
            "a3560f28-f7eb-47e2-abf5-d4bc0b9a7429/file_downloaded"
        ),
        "archive_size": 563_738_444,
        "archive_sha256": (
            "acaf808950be10a918d54c3046528b2bd2b50e5295fca5bcfba9bba564541afa"
        ),
    },
    {
        "id": "eggplant",
        "title": "Eggplant Leaf Disease Detection Dataset",
        "doi": "10.17632/d3ypkphghb.2",
        "url": "https://data.mendeley.com/datasets/d3ypkphghb/2",
        "license": "CC BY 4.0",
        "attribution": (
            "Shakib Howlader, Md. Sabbir Ahamed, and Mayen Uddin Mojumdar, "
            "Mendeley Data, DOI 10.17632/d3ypkphghb.2"
        ),
        "archive_url": (
            "https://data.mendeley.com/public-files/datasets/d3ypkphghb/files/"
            "539ff993-0ed3-431d-a4c0-b46c81b160be/file_downloaded"
        ),
        "archive_size": 2_312_791_902,
        "archive_sha256": (
            "396432e7e02ef7be84ad32eee5376a43e9a79e436585fe6db8bc29b7dbe9f356"
        ),
    },
    {
        "id": "multicrop",
        "title": "Leaf Image Dataset for Plant disease Classification and Agricultural Analysis",
        "doi": "10.17632/hgshn2zg3t.1",
        "url": "https://data.mendeley.com/datasets/hgshn2zg3t/1",
        "license": "CC BY 4.0",
        "attribution": (
            "Dip Kumar Saha and Tanjila Ferdous Sraboni, Mendeley Data, "
            "DOI 10.17632/hgshn2zg3t.1"
        ),
        "archive_url": (
            "https://data.mendeley.com/public-files/datasets/hgshn2zg3t/files/"
            "ffa4a50a-7174-4867-a372-a841dedb5767/file_downloaded"
        ),
        "archive_size": 9_914_690,
        "archive_sha256": (
            "904ca6c1f49dbb3833f62bc29ce5e39147e9e465e2a563b371dcb015336b48e8"
        ),
    },
]


class RangeFile(io.RawIOBase):
    def __init__(self, url: str, size: int) -> None:
        self.url = url
        self.size = size
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.request_count = 0
        self.bytes_requested = 0

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
            self.position = self.size + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, max(0, self.size - self.position))
        end = self.position + size
        chunks = []

        while self.position < end:
            block_start = self.position // BLOCK_SIZE * BLOCK_SIZE
            block = self.cache.get(block_start)
            if block is None:
                block_end = min(self.size - 1, block_start + BLOCK_SIZE - 1)
                expected_range = (
                    f"bytes {block_start}-{block_end}/{self.size}"
                )
                for attempt in range(3):
                    request = Request(
                        self.url,
                        headers={
                            "Range": f"bytes={block_start}-{block_end}",
                            "Accept-Encoding": "identity",
                            "Cache-Control": "no-cache",
                            "User-Agent": "Mozilla/5.0",
                        },
                    )
                    with urlopen(request, timeout=120) as response:
                        content_range = response.headers.get("Content-Range")
                        if (
                            response.status == 206
                            and content_range == expected_range
                        ):
                            block = response.read()
                            break
                        status = response.status
                    if attempt < 2:
                        time.sleep(0.5 * (attempt + 1))
                else:
                    raise OSError(
                        f"Archive range request failed at byte {block_start}: "
                        f"expected HTTP 206 with Content-Range "
                        f"{expected_range}, got HTTP {status} "
                        f"with Content-Range {content_range}"
                    )
                if len(block) != block_end - block_start + 1:
                    raise OSError(f"Incomplete archive range at byte {block_start}")
                self.request_count += 1
                self.bytes_requested += len(block)
                self.cache[block_start] = block
                if len(self.cache) > CACHE_BLOCKS:
                    self.cache.popitem(last=False)
            else:
                self.cache.move_to_end(block_start)

            offset = self.position - block_start
            count = min(end - self.position, len(block) - offset)
            if count <= 0:
                break
            chunks.append(block[offset : offset + count])
            self.position += count
        return b"".join(chunks)


def existing_image_hashes(root: Path) -> set[str]:
    hashes = set()
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for split in ("train", "val", "test"):
        folder = root / split
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in image_extensions:
                digest = hashlib.sha256()
                with path.open("rb") as image_file:
                    for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                        digest.update(chunk)
                hashes.add(digest.hexdigest())
    return hashes


def source_label(source_id: str, filename: str) -> tuple[str, str] | None:
    parts = filename.split("/")
    if source_id == "cucumber":
        if len(parts) < 3 or parts[0] != "Original Image":
            return None
        label = parts[1]
        leaf_classes = {
            "Anthracnose",
            "Bacterial Wilt",
            "Downy Mildew",
            "Fresh Leaf",
            "Gummy Stem Blight",
        }
        return ("Pepino", label) if label in leaf_classes else None
    if source_id == "eggplant":
        if len(parts) < 4 or parts[:2] != ["Eggplant Dataset", "Classified Images"]:
            return None
        return "Berinjela", parts[2]
    if source_id == "multicrop":
        if len(parts) < 4 or parts[0] != "resize plant_leaves dataset(1545)":
            return None
        if parts[1].startswith("Bitter gourd"):
            crop = "Melão-amargo"
        elif parts[1].startswith("Brinjal"):
            crop = "Berinjela"
        else:
            return None
        label = parts[2].rsplit("(", 1)[0].strip()
        return crop, label
    return None


def is_healthy(label: str) -> bool:
    normalized = " ".join(re.findall(r"[a-z]+", label.casefold()))
    tokens = set(normalized.split())
    return (
        "healthy" in tokens
        or "healty" in tokens
        or "fresh leaf" in normalized
        or "disease free" in normalized
    )


def health_metrics(rows: list[dict[str, object]]) -> dict[str, object] | None:
    if not rows:
        return None
    true_healthy = sum(bool(row["true_healthy"]) for row in rows)
    diseased = len(rows) - true_healthy
    tp = sum(row["predicted_healthy"] and row["true_healthy"] for row in rows)
    tn = sum(not row["predicted_healthy"] and not row["true_healthy"] for row in rows)
    fp = sum(not row["predicted_healthy"] and row["true_healthy"] for row in rows)
    fn = sum(row["predicted_healthy"] and not row["true_healthy"] for row in rows)
    return {
        "n": len(rows),
        "accuracy": (tp + tn) / len(rows),
        "healthy_false_positive_rate": fp / true_healthy if true_healthy else None,
        "disease_miss_rate": fn / diseased if diseased else None,
        "tp_diseased": tn,
        "fp_healthy_as_unhealthy": fp,
        "tn_healthy": tp,
        "fn_unhealthy_as_healthy": fn,
    }


def save_report(source_reports: list[dict[str, object]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "model_checkpoint": str(MODEL_PATH.resolve()),
                "confidence_threshold": 0.70,
                "sources": source_reports,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=[str(source["id"]) for source in SOURCES],
        help="conjuntos a avaliar (padrão: todos)",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        help="limita a avaliação a uma amostra estratificada por classe",
    )
    args = parser.parse_args()
    if args.max_per_class is not None and args.max_per_class <= 0:
        parser.error("--max-per-class deve ser maior que zero.")
    selected_sources = set(args.sources) if args.sources else {
        str(source["id"]) for source in SOURCES
    }

    print("Loading model and hashing existing images...", flush=True)
    device = dispositivo()
    model, classes = carregar_checkpoint(MODEL_PATH, device)
    class_metadata = json.loads(
        (DATA_ROOT / "classes.json").read_text(encoding="utf-8")
    )
    transform = transform_avaliacao()
    crop_indices: dict[str, list[int]] = defaultdict(list)
    for index, class_name in enumerate(classes):
        crop = class_metadata.get(class_name, {}).get("planta")
        if crop:
            crop_indices[crop].append(index)

    training_hashes = existing_image_hashes(DATA_ROOT)
    earlier_source_hashes: set[str] = set()
    source_reports = []
    print(f"Found {len(training_hashes)} distinct local image hashes.", flush=True)

    for source in SOURCES:
        if source["id"] not in selected_sources:
            continue
        print(f"\nEvaluating {source['id']}...", flush=True)
        remote = RangeFile(source["archive_url"], source["archive_size"])
        records: list[dict[str, object]] = []
        corrupt_images = []
        training_duplicates = 0
        source_duplicates = 0
        source_counts: Counter[tuple[str, str]] = Counter()
        pending_images: list[torch.Tensor] = []
        pending_metadata: list[dict[str, str]] = []

        def evaluate_batch() -> None:
            if not pending_images:
                return
            with torch.no_grad():
                probabilities = torch.softmax(
                    model(torch.stack(pending_images).to(device)), dim=1
                ).cpu()

            for row, item in zip(probabilities, pending_metadata):
                crop = item["crop"]
                true_label = item["label"]
                true_healthy = is_healthy(true_label)
                top1_index = int(row.argmax().item())
                crop_scores = {
                    name: float(row[indices].sum().item())
                    for name, indices in crop_indices.items()
                }
                predicted_crop = max(crop_scores, key=crop_scores.get)
                selected_index = max(
                    crop_indices[crop], key=lambda index: float(row[index].item())
                )
                selected_confidence = float(row[selected_index].item())
                crop_mismatch = predicted_crop != crop
                uncertain = crop_mismatch or selected_confidence < 0.70
                api_status = (
                    None
                    if uncertain
                    else (
                        "healthy"
                        if class_metadata[classes[selected_index]]["saudavel"]
                        else "diseased"
                    )
                )
                records.append(
                    {
                        "file": item["file"],
                        "crop": crop,
                        "label": true_label,
                        "true_healthy": true_healthy,
                        "top1_class": classes[top1_index],
                        "top1_crop": class_metadata.get(
                            classes[top1_index], {}
                        ).get("planta"),
                        "predicted_crop_by_aggregate": predicted_crop,
                        "selected_crop_class": classes[selected_index],
                        "selected_crop_confidence": selected_confidence,
                        "crop_mismatch": crop_mismatch,
                        "api_uncertain": uncertain,
                        "api_status": api_status,
                        "predicted_healthy": bool(
                            class_metadata[classes[top1_index]]["saudavel"]
                        ),
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

        with zipfile.ZipFile(remote) as archive:
            image_infos = [
                info
                for info in archive.infolist()
                if not info.is_dir()
                and info.filename.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".webp")
                )
            ]
            eligible_infos = []
            for info in image_infos:
                parsed = source_label(source["id"], info.filename)
                if parsed is None:
                    continue
                crop, label = parsed
                source_counts[(crop, label)] += 1
                eligible_infos.append((info, crop, label))
            if args.max_per_class is None:
                selected_infos = eligible_infos
            else:
                grouped_infos: dict[
                    tuple[str, str], list[tuple[zipfile.ZipInfo, str, str]]
                ] = defaultdict(list)
                for item in eligible_infos:
                    grouped_infos[(item[1], item[2])].append(item)
                rng = random.Random(42)
                selected_infos = []
                for group in sorted(grouped_infos):
                    group_infos = grouped_infos[group]
                    rng.shuffle(group_infos)
                    selected_infos.extend(group_infos[: args.max_per_class])
            if not selected_infos:
                raise ValueError(f"No original leaf images found for {source['id']}")

            for number, (info, crop, label) in enumerate(selected_infos, 1):
                raw_image = archive.read(info)
                digest = hashlib.sha256(raw_image).hexdigest()
                if digest in training_hashes:
                    training_duplicates += 1
                    continue
                if digest in earlier_source_hashes:
                    source_duplicates += 1
                    continue
                earlier_source_hashes.add(digest)
                try:
                    with Image.open(io.BytesIO(raw_image)) as image:
                        tensor = transform(image.convert("RGB"))
                except UnidentifiedImageError as error:
                    corrupt_images.append(
                        {"file": info.filename, "error": str(error)}
                    )
                    continue
                pending_images.append(tensor)
                pending_metadata.append(
                    {"file": info.filename, "crop": crop, "label": label}
                )
                if len(pending_images) >= 64:
                    evaluate_batch()
                if number % 200 == 0:
                    print(
                        f"  {number}/{len(selected_infos)} selected; "
                        f"scored={len(records) + len(pending_images)}; "
                        f"archive bytes fetched={remote.bytes_requested}",
                        flush=True,
                    )
            evaluate_batch()

        by_crop = {}
        for crop in sorted({str(row["crop"]) for row in records}):
            rows = [row for row in records if row["crop"] == crop]
            healthy_rows = [row for row in rows if row["true_healthy"]]
            diseased_rows = [row for row in rows if not row["true_healthy"]]
            accepted_rows = [row for row in rows if not row["api_uncertain"]]
            accepted_health_rows = [
                {
                    "true_healthy": row["true_healthy"],
                    "predicted_healthy": row["api_status"] == "healthy",
                }
                for row in accepted_rows
            ]
            by_crop[crop] = {
                "n": len(rows),
                "source_class_counts": {
                    label: sum(row["label"] == label for row in rows)
                    for label in sorted({str(row["label"]) for row in rows})
                },
                "raw_top1_crop_accuracy": sum(
                    row["top1_crop"] == crop for row in rows
                )
                / len(rows),
                "crop_mass_accuracy": sum(
                    row["predicted_crop_by_aggregate"] == crop for row in rows
                )
                / len(rows),
                "global_top1_health": health_metrics(rows),
                "selected_crop_coverage": len(accepted_rows) / len(rows),
                "selected_crop_health_when_accepted": health_metrics(
                    accepted_health_rows
                ),
                "selected_crop_api_false_disease_rate_on_healthy": (
                    sum(row["api_status"] == "diseased" for row in healthy_rows)
                    / len(healthy_rows)
                    if healthy_rows
                    else None
                ),
                "selected_crop_api_disease_detection_rate": (
                    sum(row["api_status"] == "diseased" for row in diseased_rows)
                    / len(diseased_rows)
                    if diseased_rows
                    else None
                ),
            }

        report = {
            "dataset": source["title"],
            "doi": source["doi"],
            "url": source["url"],
            "license": source["license"],
            "attribution": source["attribution"],
            "published_archive_sha256": source["archive_sha256"],
            "archive_size_bytes": source["archive_size"],
            "protocol": (
                "Original/classified raw images only; augmented files excluded. "
                "Exact duplicates with local train/validation/test images or earlier "
                "Mendeley sources excluded. Cucumber fruit-related classes and "
                "fresh-cucumber images excluded."
            ),
            "archive_bytes_fetched": remote.bytes_requested,
            "range_requests": remote.request_count,
            "max_per_source_class": args.max_per_class,
            "exact_duplicates_with_training_excluded": training_duplicates,
            "exact_duplicates_with_earlier_sources_excluded": source_duplicates,
            "corrupt_images": corrupt_images,
            "evaluated_rows": len(records),
            "source_class_counts_before_deduplication": {
                f"{crop} / {label}": count
                for (crop, label), count in source_counts.items()
            },
            "by_crop": by_crop,
            "notable_health_errors": sorted(
                [
                    row
                    for row in records
                    if (
                        row["true_healthy"]
                        and row["api_status"] == "diseased"
                    )
                    or (
                        not row["true_healthy"]
                        and row["api_status"] == "healthy"
                    )
                ],
                key=lambda row: row["selected_crop_confidence"],
                reverse=True,
            )[:30],
        }
        source_reports.append(report)
        save_report(source_reports)
        print(
            json.dumps(
                {
                    "source": source["id"],
                    "evaluated": len(records),
                    "duplicates_with_training": training_duplicates,
                    "duplicates_with_prior_sources": source_duplicates,
                    "archive_bytes_fetched": remote.bytes_requested,
                    "by_crop": by_crop,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    print(f"\nReport saved to {OUTPUT_PATH.resolve()}", flush=True)


if __name__ == "__main__":
    main()
