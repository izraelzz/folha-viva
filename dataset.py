from collections import Counter
import json
from typing import Literal
# Importa "Literal", usado para criar um tipo que só aceita valores específicos (ex.: só "train", "val" ou "test").

import config
# Importa o arquivo config.py (o código anterior), para usar os caminhos e hiperparâmetros definidos lá.

from torch.utils.data import DataLoader, WeightedRandomSampler
# DataLoader é a classe do PyTorch que entrega os dados em lotes (batches),
# podendo embaralhar e carregar em paralelo.

from torchvision import datasets, transforms
# torchvision é a extensão do PyTorch para visão computacional.
# datasets: classes prontas para carregar conjuntos de imagens.
# transforms: operações de pré-processamento e aumento de dados (data augmentation).

# ------------------------------------------------------------
# TIPO PERSONALIZADO PARA AS DIVISÕES DO DATASET
# ------------------------------------------------------------

Split = Literal["train", "val", "test"]
# Cria um "apelido" de tipo chamado Split que só aceita esses três textos:
#   "train" -> conjunto de treino (o modelo aprende com ele)
#   "val"   -> conjunto de validação (ajustar hiperparâmetros e acompanhar o treino)
#   "test"  -> conjunto de teste (avaliação final, usado só no fim)
# Isso ajuda editores de código e verificadores de tipo a apontar erros de digitação, como "trian".

# ------------------------------------------------------------
# TRANSFORMAÇÕES PARA O TREINO (COM DATA AUGMENTATION)
# ------------------------------------------------------------

def transform_treino() -> transforms.Compose:
    # Função que monta a sequência de transformações aplicadas às imagens de TREINO.
    # Retorna um transforms.Compose, que é uma "pipeline" de transformações em ordem.

    return transforms.Compose([
        # Compose recebe uma lista de transformações e as aplica uma após a outra.

        transforms.RandomResizedCrop(config.TAMANHO_IMAGEM, scale=(0.7, 1.0)),
        # Recorta uma região aleatória da imagem e redimensiona para 224x224.
        # scale=(0.7, 1.0): o recorte cobre entre 70% e 100% da área original.
        # Simula o objeto aparecendo em posições e "zooms" diferentes.

        transforms.RandomHorizontalFlip(),
        # Espelha a imagem horizontalmente (esquerda <-> direita) com 50% de chance.

        transforms.RandomVerticalFlip(),
        # Espelha a imagem verticalmente (cima <-> baixo) com 50% de chance.
        # Faz sentido quando a orientação não importa (ex.: imagens de microscópio, satélite, folhas).
        # Para fotos comuns (pessoas, carros), pode gerar imagens irreais.

        transforms.RandomRotation(20),
        # Gira a imagem num ângulo aleatório entre -20° e +20°.

        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        # Varia aleatoriamente brilho, contraste e saturação em até ±20%.
        # Torna o modelo mais robusto a mudanças de iluminação e câmera.

        transforms.ToTensor(),
        # Converte a imagem (formato PIL) em tensor do PyTorch.
        # Muda o formato de (altura, largura, canais) para (canais, altura, largura)
        # e escala os pixels de 0–255 para 0.0–1.0.

        transforms.Normalize(config.MEDIA, config.DESVIO),
        # Normaliza cada canal (R, G, B): (valor - média) / desvio,
        # usando as estatísticas do ImageNet definidas no config.
        # Deve vir DEPOIS do ToTensor, pois opera sobre tensores.
    ])

# Por que usar data augmentation? A cada época o modelo vê versões levemente
# diferentes das mesmas imagens, o que reduz o overfitting (decorar o treino)
# e melhora a generalização para imagens novas.

# ------------------------------------------------------------
# TRANSFORMAÇÕES PARA VALIDAÇÃO E TESTE (SEM ALEATORIEDADE)
# ------------------------------------------------------------

def transform_avaliacao() -> transforms.Compose:
    # Transformações para validação e teste. Aqui NÃO há nada aleatório:
    # queremos avaliar o modelo sempre nas mesmas condições, de forma reproduzível.

    return transforms.Compose([
        transforms.Resize(int(config.TAMANHO_IMAGEM * 1.14)),
        # Redimensiona a imagem para int(224 * 1.14) = 255 pixels no lado menor
        # (mantendo a proporção). O fator 1.14 vem da convenção 256/224 ≈ 1.14,
        # padrão usado ao avaliar modelos treinados no ImageNet.

        transforms.CenterCrop(config.TAMANHO_IMAGEM),
        # Recorta o centro da imagem em 224x224, o tamanho que o modelo espera.
        # Redimensionar um pouco maior e depois recortar o centro remove as bordas.

        transforms.ToTensor(),
        # Converte para tensor e escala os pixels para 0.0–1.0 (igual ao treino).

        transforms.Normalize(config.MEDIA, config.DESVIO),
        # Mesma normalização do treino. É essencial que treino e avaliação
        # usem a mesma normalização, senão o modelo recebe dados "diferentes".
    ])

# ------------------------------------------------------------
# CRIAÇÃO DO DATASET E DO DATALOADER
# ------------------------------------------------------------

def criar_loader(
    split: Split,
    # Qual divisão carregar: "train", "val" ou "test".

    batch_size: int = config.BATCH_SIZE,
    # Tamanho do lote; se não for informado, usa o valor do config (32).

    num_workers: int = 2,
    # Número de processos paralelos para carregar as imagens do disco.
    # Mais workers = carregamento mais rápido (até o limite de núcleos da CPU).
    # No Windows/Jupyter, às vezes é preciso usar 0 para evitar erros.

    balance_health_groups: bool = False,

) -> tuple[datasets.ImageFolder, DataLoader]:
    # A função retorna uma tupla com dois objetos: o dataset e o DataLoader.

    if balance_health_groups and split != "train":
        raise ValueError("O balanceamento de saúde só pode ser usado no treino.")

    tf: transforms.Compose = transform_treino() if split == "train" else transform_avaliacao()
    # Escolhe as transformações: se for treino, usa as com data augmentation;
    # caso contrário (val ou test), usa as transformações fixas de avaliação.
    # Isso é um "if ternário": valor_se_verdadeiro if condição else valor_se_falso.

    ds: datasets.ImageFolder = datasets.ImageFolder(config.DIR_DADOS / split, transform=tf)
    # Cria o dataset a partir da pasta data/<split> (ex.: data/train).
    # ImageFolder espera esta organização, com UMA SUBPASTA POR CLASSE:
    #   data/train/gato/img1.jpg
    #   data/train/gato/img2.jpg
    #   data/train/cachorro/img3.jpg
    # O nome de cada subpasta vira o rótulo (classe), numerado em ordem alfabética
    # (ex.: cachorro = 0, gato = 1). O mapeamento fica em ds.class_to_idx.
    # transform=tf: as transformações são aplicadas a cada imagem quando ela é lida.

    sampler = None
    if balance_health_groups:
        metadata_path = config.DIR_DADOS / "classes.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Metadados de saúde não encontrados: {metadata_path}"
            )
        metadata: dict[str, dict[str, str | bool]] = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        groups_by_target: dict[int, tuple[str, bool]] = {}
        for class_name, target in ds.class_to_idx.items():
            class_metadata = metadata.get(class_name)
            if class_metadata is None:
                raise ValueError(f"Metadados ausentes para a classe {class_name}.")
            crop = class_metadata.get("planta")
            healthy = class_metadata.get("saudavel")
            if not isinstance(crop, str) or not isinstance(healthy, bool):
                raise ValueError(f"Metadados inválidos para a classe {class_name}.")
            groups_by_target[target] = (crop, healthy)

        sample_groups = [groups_by_target[target] for target in ds.targets]
        counts = Counter(sample_groups)
        sample_weights = [1.0 / counts[group] for group in sample_groups]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(ds),
            replacement=True,
        )

    loader: DataLoader = DataLoader(
        ds,
        # O dataset de onde os dados serão retirados.

        batch_size=batch_size,
        # Quantas imagens por lote.

        shuffle=(split == "train" and not balance_health_groups),
        # Embaralha a ordem das imagens apenas no treino (True se split == "train").
        # Embaralhar evita que o modelo aprenda a ordem dos dados.
        # Em validação/teste não é necessário, e manter a ordem facilita a análise.

        num_workers=num_workers,
        # Processos paralelos usados para carregar os lotes.

        sampler=sampler,
    )

    return ds, loader
    # Retorna o dataset (útil para saber classes e quantidade de imagens)
    # e o DataLoader (usado no laço de treino/avaliação).