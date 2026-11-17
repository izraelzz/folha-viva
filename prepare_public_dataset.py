"""Prepare the CC BY 4.0 Project-AgML leaf dataset for model training."""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import TypedDict

from datasets import Image, load_dataset

import config

DATASET_ID = "Project-AgML/plant_leaf_disease_classification"
DATASET_URL = "https://huggingface.co/datasets/Project-AgML/plant_leaf_disease_classification"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DATA_DIR = config.RAIZ / "data_public"
SEED = 42

PLANT_NAMES_PT = {
    "bitter gourd": "Melão-amargo",
    "bottle gourd": "Cabaça",
    "cauliflower": "Couve-flor",
    "cucumber": "Pepino",
    "eggplant": "Berinjela",
    "tomato": "Tomate",
}

CONDITION_NAMES_PT = {
    "anthracnose": "Antracnose",
    "anthracnose lesions": "Lesões de antracnose",
    "black rot": "Podridão negra",
    "downey mildew": "Míldio",
    "downy mildew": "Míldio",
    "eggplant cercopora leaf spot": "Mancha foliar de Cercospora",
    "eggplant begomovirus": "Begomovirose",
    "eggplant fresh leaf": "Folha saudável",
    "eggplant verticillium wilt": "Murcha de Verticillium",
    "fresh leaf": "Folha saudável",
    "fusarium wilt": "Murcha de Fusarium",
    "mosaic virus": "Virose do mosaico",
    "tomato bacterial spot": "Mancha bacteriana",
    "tomato fresh leaf": "Folha saudável",
    "tomato leaf curl virus": "Virose do enrolamento das folhas",
    "tomato spotted wilt": "Virose da murcha manchada",
}


class ImageRecord(TypedDict):
    index: int
    class_id: str
    digest: str
    extension: str


def slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "_", ascii_value).strip("_")


def main() -> None:
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        raise FileExistsError(
            f"{DATA_DIR} já contém arquivos. Mova ou remova essa pasta antes de prepará-la novamente."
        )
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_ID, split="train")
    dataset = dataset.cast_column("image", Image(decode=False))

    plant_names = dataset.features["plant_type"].names
    condition_names = dataset.features["label"].names
    by_class: dict[str, list[ImageRecord]] = defaultdict(list)
    metadata: dict[str, dict[str, str | bool]] = {}
    seen: dict[str, str] = {}
    conflicting_digests: set[str] = set()
    duplicate_count = 0

    for index, row in enumerate(dataset):
        image_data = row["image"]
        content = image_data.get("bytes")
        source_path = image_data.get("path")
        if not content or not source_path:
            raise ValueError(f"Imagem ausente ou sem caminho na linha {index}.")

        plant_name = plant_names[row["plant_type"]]
        condition_name = condition_names[row["label"]]
        class_id = f"{slug(plant_name)}__{slug(condition_name)}"
        digest = hashlib.sha256(content).hexdigest()

        previous_class = seen.get(digest)
        if previous_class is not None:
            duplicate_count += 1
            if previous_class != class_id:
                conflicting_digests.add(digest)
            continue
        seen[digest] = class_id

        extension = Path(source_path).suffix.lower()
        if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
            extension = ".jpg"

        healthy = "fresh leaf" in condition_name.casefold() or "healthy" in condition_name.casefold()
        by_class[class_id].append(
            {
                "index": index,
                "class_id": class_id,
                "digest": digest,
                "extension": extension,
            }
        )
        metadata[class_id] = {
            "planta": PLANT_NAMES_PT[plant_name.casefold()],
            "condicao": CONDITION_NAMES_PT.get(condition_name.casefold(), condition_name),
            "planta_origem": plant_name,
            "condicao_origem": condition_name,
            "saudavel": healthy,
        }

    if conflicting_digests:
        for class_id in list(by_class):
            by_class[class_id] = [
                record
                for record in by_class[class_id]
                if record["digest"] not in conflicting_digests
            ]

    counts: dict[str, dict[str, int]] = {}
    rng = random.Random(SEED)
    dataset = dataset.cast_column("image", Image(decode=False))

    for class_id, records in sorted(by_class.items()):
        records = [record for record in records if record["digest"] not in conflicting_digests]
        if len(records) < 30:
            raise ValueError(
                f"A classe {class_id} tem apenas {len(records)} imagens únicas; "
                "não há dados suficientes para uma divisão segura."
            )

        rng.shuffle(records)
        train_end = int(len(records) * 0.70)
        validation_end = int(len(records) * 0.85)
        partitions = {
            "train": records[:train_end],
            "val": records[train_end:validation_end],
            "test": records[validation_end:],
        }
        counts[class_id] = {}

        for split, split_records in partitions.items():
            output_dir = DATA_DIR / split / class_id
            output_dir.mkdir(parents=True, exist_ok=True)
            counts[class_id][split] = len(split_records)
            for image_number, record in enumerate(split_records):
                image_data = dataset[record["index"]]["image"]
                content = image_data["bytes"]
                if hashlib.sha256(content).hexdigest() != record["digest"]:
                    raise ValueError(f"Hash de imagem mudou na linha {record['index']}.")
                output = output_dir / f"{image_number:05d}{record['extension']}"
                output.write_bytes(content)

    (DATA_DIR / "classes.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (DATA_DIR / "split_counts.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Dataset preparado em: {DATA_DIR}")
    print(f"Classes: {len(counts)} | imagens duplicadas removidas: {duplicate_count}")
    print(f"Imagens por divisão: {sum(item['train'] for item in counts.values())} treino, "
          f"{sum(item['val'] for item in counts.values())} validação, "
          f"{sum(item['test'] for item in counts.values())} teste")


if __name__ == "__main__":
    main()
