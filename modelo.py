from pathlib import Path
# Path: classe para manipular caminhos de arquivos (usada no tipo do parâmetro "caminho").

from typing import Any, Literal, TypedDict
# Any: indica "qualquer tipo".
# Literal: cria um tipo que só aceita valores específicos.
# TypedDict: define um dicionário com chaves fixas e tipos definidos para cada chave.

import torch
# PyTorch, a biblioteca principal de deep learning.

import torch.nn as nn
# torch.nn contém os "blocos de construção" das redes neurais: camadas, funções de ativação etc.
# É costume importá-lo com o apelido "nn".

from torchvision import models
# models: arquiteturas prontas de visão computacional (ResNet, MobileNet...),
# com opção de baixar pesos pré-treinados no ImageNet.

# ------------------------------------------------------------
# TIPOS AUXILIARES
# ------------------------------------------------------------

Arquitetura = Literal["mobilenet", "resnet18", "cnn"]
# Tipo que só aceita os três nomes de arquitetura suportados pelo projeto.

class Checkpoint(TypedDict):
    # Descreve o formato do dicionário salvo em disco ao fim do treino (o "checkpoint").
    # Não cria nada em tempo de execução; serve para documentar e ajudar o editor de código.

    arquitetura: Arquitetura
    # Qual arquitetura foi treinada (para recriar a mesma estrutura ao carregar).

    classes: list[str]
    # Lista com os nomes das classes, na ordem dos índices (ex.: ['cachorro', 'gato']).

    state_dict: dict[str, Any]
    # Os pesos aprendidos pela rede: um dicionário "nome da camada -> tensor de pesos".

# ------------------------------------------------------------
# CNN SIMPLES CRIADA DO ZERO
# ------------------------------------------------------------

class CNNSimples(nn.Module):
    # Define uma rede neural convolucional própria.
    # Toda rede no PyTorch herda de nn.Module.

    def __init__(self, num_classes: int = 2) -> None:
        # Construtor: aqui são criadas as camadas da rede.
        # num_classes: quantas classes a rede vai distinguir (padrão: 2).

        super().__init__()
        # Chama o construtor de nn.Module. É OBRIGATÓRIO: sem isso o PyTorch
        # não consegue registrar as camadas e seus parâmetros.

        def bloco(c_in: int, c_out: int) -> nn.Sequential:
            # Função auxiliar que cria um "bloco" convolucional reutilizável.
            # c_in: canais de entrada; c_out: canais de saída (número de filtros).

            return nn.Sequential(
                # nn.Sequential executa as camadas em sequência, uma após a outra.

                nn.Conv2d(c_in, c_out, 3, padding=1),
                # Convolução 2D com c_out filtros de tamanho 3x3.
                # Cada filtro aprende a detectar um padrão (bordas, texturas, formas...).
                # padding=1 adiciona uma borda de 1 pixel, mantendo altura e largura iguais.

                nn.BatchNorm2d(c_out),
                # Normalização em lote: padroniza as saídas de cada canal.
                # Deixa o treino mais estável e rápido.

                nn.ReLU(inplace=True),
                # Função de ativação ReLU: troca valores negativos por 0.
                # Introduz não linearidade, permitindo à rede aprender padrões complexos.
                # inplace=True altera o tensor diretamente, economizando memória.

                nn.MaxPool2d(2),
                # Max pooling 2x2: em cada janela 2x2, mantém apenas o maior valor.
                # Reduz altura e largura pela metade (ex.: 224 -> 112).
            )

        self.features: nn.Sequential = nn.Sequential(
            # Parte da rede que EXTRAI características da imagem.

            bloco(3, 32), bloco(32, 64), bloco(64, 128), bloco(128, 256),
            # Quatro blocos em sequência. Os canais aumentam (3 -> 32 -> 64 -> 128 -> 256)
            # enquanto o tamanho espacial diminui. Para uma entrada 224x224:
            #   entrada:  3 x 224 x 224  (RGB)
            #   bloco 1: 32 x 112 x 112
            #   bloco 2: 64 x  56 x  56
            #   bloco 3: 128 x 28 x  28
            #   bloco 4: 256 x 14 x  14
            # Camadas iniciais detectam padrões simples; as finais, padrões mais complexos.

            nn.AdaptiveAvgPool2d(1),
            # Calcula a média de cada canal, reduzindo 256 x 14 x 14 para 256 x 1 x 1.
            # "Adaptive" significa que funciona com qualquer tamanho de entrada.
        )

        self.classificador: nn.Sequential = nn.Sequential(
            # Parte da rede que DECIDE a classe a partir das características extraídas.

            nn.Flatten(),
            # "Achata" o tensor: de (batch, 256, 1, 1) para (batch, 256).

            nn.Dropout(0.3),
            # Durante o treino, zera aleatoriamente 30% dos valores.
            # Força a rede a não depender de poucos neurônios, reduzindo overfitting.
            # Em modo de avaliação (.eval()) o dropout fica desligado.

            nn.Linear(256, num_classes)
            # Camada totalmente conectada: transforma 256 valores em num_classes valores.
            # A saída são os "logits" (pontuações brutas de cada classe, ainda sem softmax).
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Define o caminho dos dados pela rede (a "passagem para frente").
        # O PyTorch chama este método quando fazemos modelo(imagens).

        return self.classificador(self.features(x))
        # Primeiro extrai as características, depois classifica.

# ------------------------------------------------------------
# FÁBRICA DE MODELOS
# ------------------------------------------------------------

def criar_modelo(
    arquitetura: Arquitetura = "mobilenet",
    # Qual arquitetura criar (padrão: MobileNet).

    num_classes: int = 2,
    # Número de classes do problema.

    pretreinado: bool = True,
    # Se True, carrega pesos já treinados no ImageNet (transfer learning).

) -> nn.Module:
    # Retorna o modelo pronto (qualquer rede é um nn.Module).

    if arquitetura == "cnn":
        return CNNSimples(num_classes)
        # A CNN própria não tem pesos pré-treinados: sempre começa do zero.

    if arquitetura == "mobilenet":
        # MobileNetV3 Small: rede leve e rápida, pensada para celulares e dispositivos fracos.

        pesos_mb: models.MobileNet_V3_Small_Weights | None = (
            models.MobileNet_V3_Small_Weights.DEFAULT if pretreinado else None
        )
        # Escolhe os pesos: DEFAULT = melhores pesos disponíveis treinados no ImageNet;
        # None = pesos aleatórios (rede começa sem conhecimento).

        mb: models.MobileNetV3 = models.mobilenet_v3_small(weights=pesos_mb)
        # Cria a MobileNetV3 Small (na primeira vez, baixa os pesos da internet).
        # Ela vem preparada para as 1000 classes do ImageNet.

        ultima = mb.classifier[3]
        # O classificador da MobileNetV3 Small é um Sequential com 4 camadas:
        # [Linear, Hardswish, Dropout, Linear]. O índice 3 é a última camada Linear,
        # que produz as 1000 saídas do ImageNet.

        assert isinstance(ultima, nn.Linear)
        # Confere que essa camada é realmente uma nn.Linear (se não for, gera erro).
        # Também ajuda o verificador de tipos a saber que "ultima" tem .in_features.

        mb.classifier[3] = nn.Linear(ultima.in_features, num_classes)
        # Substitui a última camada por uma nova com num_classes saídas.
        # in_features (1024) é mantido para encaixar com a camada anterior.
        # Esse é o passo central do transfer learning: reaproveita o que a rede
        # aprendeu e troca só a "cabeça" para o nosso problema.

        return mb

    if arquitetura == "resnet18":
        # ResNet-18: rede de 18 camadas com conexões residuais ("atalhos"),
        # que facilitam o treino de redes profundas.

        pesos_rn: models.ResNet18_Weights | None = (
            models.ResNet18_Weights.DEFAULT if pretreinado else None
        )
        # Mesmo raciocínio: pesos do ImageNet ou nenhum peso.

        rn: models.ResNet = models.resnet18(weights=pesos_rn)
        # Cria a ResNet-18.

        rn.fc = nn.Linear(rn.fc.in_features, num_classes)
        # Na ResNet, a última camada se chama "fc" (fully connected).
        # Ela é trocada por uma nova com num_classes saídas (in_features = 512).

        return rn

    raise ValueError(f"Arquitetura desconhecida: {arquitetura}")
    # Se o nome não for nenhum dos três, interrompe com uma mensagem de erro clara.

# ------------------------------------------------------------
# CARREGAR UM MODELO SALVO
# ------------------------------------------------------------

def carregar_checkpoint(
    caminho: str | Path,
    # Caminho do arquivo salvo (ex.: "modelos/melhor.pt"). Aceita texto ou Path.

    device: str | torch.device = "cpu",
    # Onde o modelo deve ficar: "cpu", "cuda" ou "mps".

) -> tuple[nn.Module, list[str]]:
    # Retorna o modelo carregado e a lista de nomes das classes.

    ckpt: Checkpoint = torch.load(caminho, map_location=device)
    # Lê o arquivo do disco e recupera o dicionário salvo.
    # map_location=device permite abrir na CPU um modelo treinado em GPU (e vice-versa).

    modelo: nn.Module = criar_modelo(ckpt["arquitetura"], len(ckpt["classes"]), pretreinado=False)
    # Recria a MESMA estrutura usada no treino, com o mesmo número de classes.
    # pretreinado=False porque os pesos virão do checkpoint; não precisa baixar os do ImageNet.

    modelo.load_state_dict(ckpt["state_dict"])
    # Copia os pesos treinados para dentro da estrutura recém-criada.
    # Se a estrutura não bater com os pesos, o PyTorch gera erro.

    modelo.to(device).eval()
    # .to(device): move o modelo para CPU/GPU.
    # .eval(): coloca em modo de avaliação, o que desliga o Dropout
    # e faz o BatchNorm usar as estatísticas aprendidas no treino.
    # Esquecer o .eval() é um erro comum que gera previsões instáveis.

    return modelo, ckpt["classes"]
    # Devolve o modelo pronto para fazer previsões e os nomes das classes,
    # para converter o índice previsto (ex.: 1) no nome (ex.: "gato").