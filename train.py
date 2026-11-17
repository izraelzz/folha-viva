import argparse
# argparse: lê argumentos passados na linha de comando
# (ex.: python train.py --epocas 20 --lr 0.0005).

import json
# json: salva dados (o histórico de treino) em arquivo de texto no formato JSON.

import time
# time: mede quanto tempo cada época leva.

from pathlib import Path
# Path: manipula caminhos de arquivos.

from typing import get_args
# get_args: extrai os valores de um tipo Literal.
# Ex.: get_args(Arquitetura) -> ("mobilenet", "resnet18", "cnn").

import matplotlib
# Biblioteca de gráficos.

matplotlib.use("Agg")
# Define o "backend" Agg, que desenha gráficos direto em arquivo, sem abrir janela.
# Útil em servidores, clusters e máquinas sem interface gráfica.
# Precisa vir ANTES de importar pyplot.

import matplotlib.pyplot as plt
# pyplot: interface do matplotlib para criar figuras e gráficos.

import torch
# PyTorch.

import torch.nn as nn
# Camadas, funções de perda e a classe base nn.Module.

from torch.utils.data import DataLoader
# Usado aqui só como tipo (anotação) do parâmetro "loader".

import config
# Arquivo de configuração (caminhos, hiperparâmetros, escolha do dispositivo).

from dataset import criar_loader
# Função que cria o dataset e o DataLoader (arquivo dataset.py).

from modelo import Arquitetura, Checkpoint, criar_modelo
# Tipos e função de criação de modelos (arquivo modelo.py).

Historico = dict[str, list[float]]
# Apelido de tipo: um dicionário em que cada chave (ex.: "perda_treino")
# guarda uma lista com um valor por época.

# ------------------------------------------------------------
# UMA ÉPOCA DE TREINO OU DE VALIDAÇÃO
# ------------------------------------------------------------

def rodar_epoca(
    modelo: nn.Module,
    # A rede neural.

    loader: DataLoader,
    # De onde vêm os lotes de imagens e rótulos.

    criterio: nn.Module,
    # Função de perda (mede o quanto o modelo errou).

    device: torch.device,
    # CPU ou GPU.

    otimizador: torch.optim.Optimizer | None = None,
    # Se for passado um otimizador, a função TREINA.
    # Se for None, apenas AVALIA (validação). Assim a mesma função serve para os dois casos.

) -> tuple[float, float]:
    # Retorna (perda média, acurácia) da época.

    treinando: bool = otimizador is not None
    # True se estamos treinando, False se estamos avaliando.

    modelo.train(treinando)
    # Coloca o modelo no modo certo:
    #   train(True)  -> modo treino (Dropout ativo, BatchNorm atualiza estatísticas)
    #   train(False) -> equivalente a modelo.eval()

    perda_total: float = 0.0
    # Soma das perdas de todas as imagens da época.

    acertos: int = 0
    # Quantas imagens o modelo classificou corretamente.

    total: int = 0
    # Quantas imagens foram processadas.

    imgs: torch.Tensor
    rotulos: torch.Tensor
    # Apenas declaram os tipos das variáveis usadas no laço (não executam nada).

    with torch.set_grad_enabled(treinando):
        # Liga o cálculo de gradientes só durante o treino.
        # Na validação ele fica desligado: economiza memória e deixa tudo mais rápido,
        # já que não vamos atualizar os pesos.

        for imgs, rotulos in loader:
            # Percorre o dataset lote por lote.
            # imgs: (batch, 3, 224, 224); rotulos: (batch,)

            imgs, rotulos = imgs.to(device), rotulos.to(device)
            # Envia os dados para o mesmo dispositivo do modelo (CPU ou GPU).
            # Modelo e dados precisam estar no mesmo lugar, senão dá erro.

            saidas: torch.Tensor = modelo(imgs)
            # Passagem para frente (forward): o modelo gera os logits,
            # um tensor (batch, num_classes) com a pontuação de cada classe.

            perda: torch.Tensor = criterio(saidas, rotulos)
            # Calcula a perda comparando as previsões com os rótulos verdadeiros.
            # O valor é a MÉDIA da perda das imagens do lote.

            if otimizador is not None:
                # Só entra aqui no treino.

                otimizador.zero_grad()
                # Zera os gradientes do lote anterior.
                # O PyTorch ACUMULA gradientes por padrão; sem zerar, eles se somariam.

                perda.backward()
                # Retropropagação (backpropagation): calcula o gradiente da perda
                # em relação a cada peso da rede.

                otimizador.step()
                # Atualiza os pesos na direção que reduz a perda, usando os gradientes.

            perda_total += perda.item() * imgs.size(0)
            # .item() converte o tensor de um valor em número Python.
            # Multiplica pelo tamanho do lote porque a perda é uma média;
            # assim somamos a perda "total" do lote. Isso deixa a média final correta
            # mesmo quando o último lote é menor que os outros.

            acertos += int((saidas.argmax(1) == rotulos).sum().item())
            # argmax(1): pega, para cada imagem, o índice da classe com maior pontuação
            # (a classe prevista).
            # == rotulos: compara com o rótulo verdadeiro (True/False por imagem).
            # .sum(): conta quantos True (acertos) houve no lote.

            total += imgs.size(0)
            # Soma a quantidade de imagens do lote.

    return perda_total / total, acertos / total
    # Perda média por imagem e acurácia (proporção de acertos, entre 0 e 1).

# ------------------------------------------------------------
# GRÁFICO DAS CURVAS DE TREINO
# ------------------------------------------------------------

def salvar_grafico(hist: Historico, caminho: Path) -> None:
    # Desenha as curvas de perda e acurácia e salva como imagem.

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    # Cria uma figura com 1 linha e 2 colunas de gráficos (a1 e a2).
    # figsize=(11, 4): tamanho em polegadas (largura, altura).

    ep: range = range(1, len(hist["perda_treino"]) + 1)
    # Eixo X: número das épocas, de 1 até a quantidade de épocas registradas.

    a1.plot(ep, hist["perda_treino"], label="treino")
    # Curva da perda no treino.

    a1.plot(ep, hist["perda_val"], label="validação")
    # Curva da perda na validação.

    a1.set_title("Perda")
    # Título do primeiro gráfico.

    a1.set_xlabel("época")
    # Rótulo do eixo X.

    a1.legend()
    # Mostra a legenda usando os "label" definidos acima.

    a2.plot(ep, hist["acc_treino"], label="treino")
    # Curva da acurácia no treino.

    a2.plot(ep, hist["acc_val"], label="validação")
    # Curva da acurácia na validação.

    a2.set_title("Acurácia")
    a2.set_xlabel("época")
    a2.legend()
    # Título, rótulo do eixo X e legenda do segundo gráfico.

    fig.tight_layout()
    # Ajusta espaçamentos para que títulos e rótulos não se sobreponham.

    fig.savefig(caminho, dpi=120)
    # Salva a figura em arquivo; dpi controla a resolução.

    plt.close(fig)
    # Fecha a figura para liberar memória.

# Como ler as curvas: se a perda de treino continua caindo mas a de validação
# começa a subir, o modelo está sofrendo overfitting (decorando o treino).

# ------------------------------------------------------------
# FUNÇÃO PRINCIPAL
# ------------------------------------------------------------

def main() -> None:

    p: argparse.ArgumentParser = argparse.ArgumentParser()
    # Cria o leitor de argumentos de linha de comando.

    p.add_argument("--arquitetura", default="mobilenet", choices=list(get_args(Arquitetura)))
    # --arquitetura: qual rede usar. Só aceita os valores do tipo Arquitetura.
    # Se for omitido, usa "mobilenet".

    p.add_argument("--epocas", type=int, default=config.EPOCAS)
    # --epocas: número de épocas (padrão vem do config).

    p.add_argument("--lr", type=float, default=config.LR)
    # --lr: taxa de aprendizado.

    p.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    # --batch: tamanho do lote.

    p.add_argument("--sem-pretreino", action="store_true", help="não baixa pesos do ImageNet (treina do zero)")
    # --sem-pretreino: é uma "flag". Se aparecer no comando, vale True; senão, False.
    # help: texto exibido ao rodar "python train.py --help".

    p.add_argument(
        "--balance-health-groups",
        action="store_true",
        help="equilibra grupos saudável/doente dentro de cada cultura",
    )
    p.add_argument(
        "--checkpoint-inicial",
        type=Path,
        help="checkpoint compatível para continuar o treino",
    )

    args: argparse.Namespace = p.parse_args()
    # Lê os argumentos digitados. Os valores ficam acessíveis como atributos,
    # ex.: args.epocas. Hífens viram underscore: --sem-pretreino -> args.sem_pretreino.

    arquitetura: Arquitetura = args.arquitetura
    epocas: int = args.epocas
    # Guarda os valores em variáveis com tipos explícitos, para facilitar a leitura.

    torch.manual_seed(config.SEED)
    # Fixa a semente aleatória (inicialização de pesos, embaralhamento...),
    # para que o experimento seja reproduzível.

    device: torch.device = config.dispositivo()
    # Escolhe automaticamente GPU NVIDIA, GPU do Mac ou CPU.

    print(f"Dispositivo: {device}")
    # Mostra qual dispositivo será usado.

    ds_tr, ld_tr = criar_loader(
        "train",
        args.batch,
        balance_health_groups=args.balance_health_groups,
    )
    # Dataset e DataLoader de treino (com data augmentation e embaralhamento).

    ds_va, ld_va = criar_loader("val", args.batch)
    # Dataset e DataLoader de validação (sem augmentation, sem embaralhar).

    classes: list[str] = ds_tr.classes
    # Nomes das classes, tirados das subpastas de data/train.

    print(f"Classes: {classes} | treino: {len(ds_tr)} | validação: {len(ds_va)}")
    # Mostra as classes e quantas imagens há em cada conjunto.

    pretreino: bool = not args.sem_pretreino and arquitetura != "cnn"
    # Usa pesos pré-treinados somente se:
    #   1) o usuário NÃO passou --sem-pretreino, E
    #   2) a arquitetura não é a "cnn" (que não tem pesos pré-treinados).

    modelo: nn.Module
    # Declara o tipo da variável que será definida logo abaixo.

    if args.checkpoint_inicial is not None:
        checkpoint_inicial: Checkpoint = torch.load(
            args.checkpoint_inicial,
            map_location="cpu",
        )
        if checkpoint_inicial["arquitetura"] != arquitetura:
            raise ValueError(
                "A arquitetura do checkpoint inicial não corresponde à arquitetura selecionada."
            )
        if checkpoint_inicial["classes"] != classes:
            raise ValueError(
                "As classes do checkpoint inicial não correspondem às classes do dataset."
            )
        modelo = criar_modelo(arquitetura, len(classes), pretreinado=False)
        modelo.load_state_dict(checkpoint_inicial["state_dict"])
    else:
        try:
            modelo = criar_modelo(arquitetura, len(classes), pretreinado=pretreino)
            # Tenta criar o modelo (pode precisar baixar pesos da internet).

        except Exception as e:  # sem internet para baixar os pesos
            print(f"Não foi possível baixar pesos pré-treinados ({e}). Treinando do zero.")
            modelo = criar_modelo(arquitetura, len(classes), pretreinado=False)
            # Se der qualquer erro (ex.: sem internet), avisa e cria o modelo sem pesos prévios,
            # para que o script não pare.

    modelo.to(device)
    # Move o modelo para o dispositivo escolhido.

    criterio: nn.Module = nn.CrossEntropyLoss()
    # Função de perda padrão para classificação com várias classes.
    # Ela já aplica o softmax internamente, por isso o modelo devolve logits "crus".

    otimizador: torch.optim.Optimizer = torch.optim.AdamW(
        modelo.parameters(), lr=args.lr, weight_decay=1e-4
    )
    # AdamW: otimizador que ajusta a taxa de aprendizado de cada peso individualmente.
    # modelo.parameters(): todos os pesos treináveis da rede.
    # weight_decay=1e-4: regularização que "puxa" os pesos para valores pequenos,
    # ajudando a evitar overfitting.

    agendador: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        otimizador, T_max=epocas
    )
    # Agendador da taxa de aprendizado: reduz o LR ao longo do treino seguindo
    # uma curva de cosseno, do valor inicial até perto de zero na última época.
    # Passos grandes no início para aprender rápido; passos pequenos no fim para refinar.

    config.DIR_MODELOS.mkdir(exist_ok=True)
    config.DIR_RESULTADOS.mkdir(exist_ok=True)
    # Cria as pastas "modelos" e "resultados". exist_ok=True: não dá erro se já existirem.

    caminho_modelo: Path = config.DIR_MODELOS / "melhor_modelo.pth"
    # Caminho onde será salvo o melhor modelo.

    hist: Historico = {k: [] for k in ["perda_treino", "acc_treino", "perda_val", "acc_val"]}
    # Cria o histórico com uma lista vazia para cada métrica (dict comprehension).

    melhor_acc: float = 0.0
    # Melhor acurácia de validação vista até agora.

    for ep in range(1, epocas + 1):
        # Laço principal: uma volta por época.

        t0: float = time.time()
        # Marca o horário de início da época.

        pt, at = rodar_epoca(modelo, ld_tr, criterio, device, otimizador)
        # Treina uma época. pt = perda de treino; at = acurácia de treino.
        # (Passa o otimizador, então a função treina.)

        pv, av = rodar_epoca(modelo, ld_va, criterio, device)
        # Avalia na validação. pv = perda de validação; av = acurácia de validação.
        # (Sem otimizador, então a função só avalia.)

        agendador.step()
        # Atualiza a taxa de aprendizado para a próxima época.

        for k, v in zip(hist, [pt, at, pv, av]):
            hist[k].append(v)
        # Percorrer um dicionário dá suas chaves, na ordem em que foram criadas:
        # perda_treino, acc_treino, perda_val, acc_val.
        # zip junta cada chave ao valor correspondente e o adiciona na lista.

        marca: str = ""
        # Texto extra para o print (fica vazio se este não for o melhor modelo).

        if av > melhor_acc:
            # Se a acurácia de validação desta época for a melhor até agora...

            melhor_acc = av
            # ...atualiza o recorde...

            ckpt: Checkpoint = {
                "state_dict": modelo.state_dict(),
                # Os pesos atuais da rede.

                "classes": classes,
                # Os nomes das classes.

                "arquitetura": arquitetura,
                # Qual arquitetura foi usada (para recriá-la depois).
            }
            # ...monta o checkpoint no formato definido em modelo.py...

            torch.save(ckpt, caminho_modelo)
            # ...e salva em disco, substituindo o anterior.
            # Essa técnica guarda o modelo da MELHOR época, não o da última,
            # protegendo contra o overfitting das épocas finais.

            marca = "  (melhor, salvo)"
            # Indica no print que esta época gerou um novo melhor modelo.

        print(f"Época {ep:2d}/{epocas} | treino perda {pt:.3f} acc {at:.1%} | "
              f"val perda {pv:.3f} acc {av:.1%} | {time.time() - t0:.0f}s{marca}")
        # Mostra o resumo da época. Formatações usadas:
        #   {ep:2d}  -> inteiro com 2 posições (alinha " 1" e "10")
        #   {pt:.3f} -> número com 3 casas decimais
        #   {at:.1%} -> converte 0.873 em "87.3%"
        #   {...:.0f} -> segundos sem casas decimais

    salvar_grafico(hist, config.DIR_RESULTADOS / "curvas_treino.png")
    # Ao final do treino, salva o gráfico com as curvas de perda e acurácia.

    with open(config.DIR_RESULTADOS / "historico.json", "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
    # Salva o histórico completo em JSON (indent=2 deixa o arquivo legível).
    # O "with" garante que o arquivo seja fechado corretamente.

    print(f"\nMelhor acurácia de validação: {melhor_acc:.1%}")
    print(f"Modelo salvo em: {caminho_modelo}")
    print("Próximo passo: python evaluate.py")
    # Resumo final e orientação sobre o próximo script a rodar.

if __name__ == "__main__":
    main()
    # Executa main() somente quando o arquivo é rodado diretamente
    # (python train.py). Se ele for importado por outro arquivo, main() não roda.