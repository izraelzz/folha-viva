import argparse
# Lê argumentos da linha de comando (ex.: python predict.py folha.jpg).

from pathlib import Path
# Manipula caminhos de arquivos e pastas.

import torch
# PyTorch.

from PIL import Image
# PIL (Pillow): biblioteca para abrir e manipular imagens.

import config
# Caminhos e escolha do dispositivo.

from dataset import transform_avaliacao
# Reaproveita as MESMAS transformações usadas na validação e no teste.
# A imagem nova precisa ser preparada exatamente como as do treino/avaliação,
# senão o modelo recebe dados num formato diferente do que aprendeu.

from modelo import carregar_checkpoint
# Recria o modelo e carrega os pesos treinados.

EXTENSOES: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Conjunto (set) de extensões aceitas como imagem.
# Usado para ignorar outros arquivos ao processar uma pasta inteira.

# ------------------------------------------------------------
# PREVISÃO DE UMA ÚNICA IMAGEM
# ------------------------------------------------------------

def prever(
    modelo: torch.nn.Module,
    # O modelo já treinado e em modo .eval().

    classes: list[str],
    # Nomes das classes, na ordem dos índices.

    caminho: Path,
    # Caminho do arquivo de imagem.

    device: torch.device,
    # CPU ou GPU.

) -> tuple[str, float]:
    # Retorna (nome da classe prevista, confiança entre 0 e 1).

    img: Image.Image = Image.open(caminho).convert("RGB")
    # Abre a imagem e converte para RGB (3 canais).
    # Necessário porque algumas imagens vêm em tons de cinza (1 canal)
    # ou PNG com transparência (4 canais, RGBA), e o modelo espera exatamente 3.

    x: torch.Tensor = transform_avaliacao()(img).unsqueeze(0).to(device)
    # Três passos encadeados:
    #   transform_avaliacao()(img) -> cria as transformações e aplica na imagem
    #                                 (redimensiona, recorta o centro, vira tensor, normaliza).
    #                                 Resultado: tensor (3, 224, 224).
    #   .unsqueeze(0)              -> adiciona uma dimensão no início: (1, 3, 224, 224).
    #                                 O modelo sempre espera um LOTE de imagens,
    #                                 então criamos um lote com uma imagem só.
    #   .to(device)                -> envia para o mesmo dispositivo do modelo.

    with torch.no_grad():
        # Sem cálculo de gradientes: estamos só prevendo, não treinando.

        probs: torch.Tensor = torch.softmax(modelo(x), dim=1)[0].cpu()
        # modelo(x)           -> logits, formato (1, num_classes)
        # torch.softmax(dim=1)-> converte os logits em probabilidades que somam 1
        #                        (ex.: [2.1, -0.4] vira [0.92, 0.08])
        # [0]                 -> pega a primeira (e única) imagem do lote: (num_classes,)
        # .cpu()              -> traz o resultado para a CPU

    idx: int = int(probs.argmax())
    # Índice da classe com maior probabilidade (a classe prevista).

    return classes[idx], float(probs[idx])
    # Devolve o nome da classe e a probabilidade dela (a "confiança").

# ------------------------------------------------------------
# FUNÇÃO PRINCIPAL
# ------------------------------------------------------------

def main() -> None:

    p: argparse.ArgumentParser = argparse.ArgumentParser()
    # Cria o leitor de argumentos.

    p.add_argument("caminho", help="imagem ou pasta de imagens")
    # Argumento POSICIONAL (sem "--"): é obrigatório e vem logo após o nome do script.
    # Pode ser um arquivo de imagem ou uma pasta.

    p.add_argument("--modelo", default=str(config.DIR_MODELOS / "melhor_modelo.pth"))
    # Argumento opcional: qual arquivo de modelo usar.
    # Padrão: o melhor modelo salvo pelo train.py.

    args: argparse.Namespace = p.parse_args()
    # Lê os argumentos digitados.

    device: torch.device = config.dispositivo()
    # Escolhe GPU NVIDIA, GPU do Mac ou CPU.

    modelo: torch.nn.Module
    classes: list[str]
    # Declaração dos tipos.

    modelo, classes = carregar_checkpoint(args.modelo, device)
    # Carrega o modelo treinado (já em modo .eval()) e os nomes das classes.

    alvo: Path = Path(args.caminho)
    # Converte o caminho digitado (texto) em objeto Path.

    arquivos: list[Path]
    # Lista de imagens que serão classificadas.

    if alvo.is_dir():
        # Se o caminho for uma PASTA...

        arquivos = sorted(f for f in alvo.rglob("*") if f.suffix.lower() in EXTENSOES)
        # rglob("*")      -> percorre a pasta e TODAS as subpastas (busca recursiva)
        # f.suffix        -> extensão do arquivo (ex.: ".JPG")
        # .lower()        -> passa para minúsculas, para aceitar ".JPG" e ".jpg"
        # in EXTENSOES    -> mantém só os arquivos de imagem
        # sorted(...)     -> ordena em ordem alfabética, para a saída ser previsível

    else:
        arquivos = [alvo]
        # Se for um ARQUIVO, a lista tem só ele.

    for f in arquivos:
        # Classifica cada imagem, uma por vez.

        classe, conf = prever(modelo, classes, f, device)
        # Obtém a classe prevista e a confiança.

        emoji: str = "🌿" if classe == "saudavel" else "🍂"
        # Escolhe um emoji: folha verde se a classe for "saudavel", folha seca caso contrário.
        # Isso depende do nome exato da pasta de classe; se ela se chamar
        # "saudável" (com acento) ou "Saudavel", sempre aparecerá 🍂.

        print(f"{emoji} {f.name}: {classe.upper()} ({conf:.1%} de confiança)")
        # Mostra o resultado. f.name é só o nome do arquivo (sem a pasta);
        # .upper() deixa a classe em maiúsculas; :.1% formata 0.923 como "92.3%".

if __name__ == "__main__":
    main()
    # Executa main() somente quando o arquivo é rodado diretamente.