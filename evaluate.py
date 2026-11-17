import matplotlib
# Biblioteca de gráficos.

matplotlib.use("Agg")
# Backend que salva gráficos direto em arquivo, sem abrir janela.
# Precisa vir ANTES de importar pyplot.

import matplotlib.pyplot as plt
# Interface para criar e salvar figuras.

import numpy as np
# NumPy: biblioteca de arrays numéricos. Aqui é usada só como tipo da matriz de confusão.

import torch
# PyTorch.

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
# Ferramentas do scikit-learn para avaliar classificadores:
#   classification_report  -> relatório com precisão, recall e F1 por classe
#   confusion_matrix       -> calcula a matriz de confusão
#   ConfusionMatrixDisplay -> desenha a matriz de confusão como imagem

import config
# Caminhos e escolha do dispositivo.

from dataset import criar_loader
# Função que cria o DataLoader (aqui, do conjunto de teste).

from modelo import carregar_checkpoint
# Função que recria o modelo e carrega os pesos salvos no treino.

# ------------------------------------------------------------
# FUNÇÃO PRINCIPAL
# ------------------------------------------------------------

def main() -> None:

    device: torch.device = config.dispositivo()
    # Escolhe GPU NVIDIA, GPU do Mac ou CPU.

    modelo: torch.nn.Module
    classes: list[str]
    # Apenas declaram os tipos das variáveis preenchidas na linha seguinte.

    modelo, classes = carregar_checkpoint(config.DIR_MODELOS / "melhor_modelo.pth", device)
    # Carrega o melhor modelo salvo pelo train.py e a lista de nomes das classes.
    # carregar_checkpoint já coloca o modelo em modo .eval().

    ds, loader = criar_loader("test")
    # Dataset e DataLoader do conjunto de TESTE (pasta data/test).
    # Usa as transformações de avaliação (sem aleatoriedade) e NÃO embaralha (shuffle=False).
    # A ordem fixa é importante mais abaixo, para descobrir quais arquivos o modelo errou.

    reais: list[int] = []
    # Guardará os rótulos verdadeiros de todas as imagens de teste.

    preds: list[int] = []
    # Guardará as previsões do modelo para cada imagem.

    imgs: torch.Tensor
    rotulos: torch.Tensor
    # Declaração dos tipos usados no laço.

    with torch.no_grad():
        # Desliga o cálculo de gradientes: na avaliação não atualizamos pesos,
        # então isso economiza memória e acelera o processamento.

        for imgs, rotulos in loader:
            # Percorre o conjunto de teste lote por lote.

            saidas: torch.Tensor = modelo(imgs.to(device))
            # Envia as imagens para o dispositivo e obtém os logits (pontuação de cada classe).

            preds += saidas.argmax(1).cpu().tolist()
            # argmax(1): índice da classe com maior pontuação em cada imagem (a previsão).
            # .cpu(): traz o resultado de volta da GPU para a CPU (necessário para converter).
            # .tolist(): transforma o tensor em lista Python.
            # "+=" adiciona esses valores ao final da lista preds.

            reais += rotulos.tolist()
            # Adiciona os rótulos verdadeiros do lote.
            # Os rótulos nunca foram enviados para a GPU, por isso não precisam de .cpu().

    # ---------------- RELATÓRIO DE MÉTRICAS ----------------

    relatorio: str = classification_report(reais, preds, target_names=classes, digits=3)
    # Gera um relatório em texto com, para cada classe:
    #   precision (precisão) -> das imagens que o modelo disse ser da classe X, quantas eram X
    #   recall (revocação)   -> das imagens que eram X de verdade, quantas o modelo encontrou
    #   f1-score             -> média harmônica entre precisão e recall
    #   support              -> quantas imagens daquela classe existem no teste
    # Também mostra a acurácia geral e médias entre as classes.
    # target_names: usa os nomes das classes no lugar de 0, 1, 2...
    # digits=3: três casas decimais.

    print(relatorio)
    # Mostra o relatório no terminal.

    config.DIR_RESULTADOS.mkdir(exist_ok=True)
    # Garante que a pasta "resultados" existe.

    (config.DIR_RESULTADOS / "relatorio_teste.txt").write_text(relatorio, encoding="utf-8")
    # Salva o relatório em arquivo de texto.
    # write_text é um atalho do Path que abre, escreve e fecha o arquivo numa linha só.

    # ---------------- MATRIZ DE CONFUSÃO ----------------

    cm: np.ndarray = confusion_matrix(reais, preds)
    # Calcula a matriz de confusão: uma tabela em que
    #   LINHAS  = classe real
    #   COLUNAS = classe prevista
    # A diagonal principal contém os acertos; fora dela, os erros.
    # Ex.: o valor na linha "gato", coluna "cachorro" = quantos gatos foram previstos como cachorro.

    ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap="Greens", colorbar=False)
    # Desenha a matriz como imagem, com os nomes das classes nos eixos.
    # cmap="Greens": escala de cores verde (mais escuro = mais imagens).
    # colorbar=False: não mostra a barra lateral de cores.

    plt.title("Matriz de confusão (teste)")
    # Título do gráfico.

    plt.tight_layout()
    # Ajusta espaçamentos para nada ficar cortado.

    plt.savefig(config.DIR_RESULTADOS / "matriz_confusao.png", dpi=120)
    # Salva a imagem da matriz de confusão.

    # ---------------- LISTA DE IMAGENS ERRADAS ----------------

    erros: list[tuple[str, str, str]] = [
        (ds.samples[i][0], classes[r], classes[p])
        for i, (r, p) in enumerate(zip(reais, preds))
        if r != p
    ]
    # List comprehension que monta a lista de erros. Lendo de baixo para cima:
    #   zip(reais, preds)  -> junta cada rótulo real com sua previsão: (r, p)
    #   enumerate(...)     -> acrescenta o índice i da imagem: i, (r, p)
    #   if r != p          -> mantém só os casos em que o modelo errou
    #   ds.samples[i][0]   -> ds.samples é uma lista de (caminho_do_arquivo, rótulo);
    #                         [i][0] pega o caminho do arquivo da i-ésima imagem
    #   classes[r], classes[p] -> convertem os índices em nomes de classe
    # Isso só funciona porque o loader de teste NÃO embaralha: a i-ésima previsão
    # corresponde exatamente à i-ésima imagem de ds.samples.

    if erros:
        # Se houver pelo menos um erro (lista não vazia)...

        print(f"\n{len(erros)} imagens classificadas errado:")
        # ...mostra quantos erros houve...

        for caminho, r, p in erros:
            print(f"  {caminho}  (real: {r}, previsto: {p})")
            # ...e lista cada arquivo errado com a classe real e a classe prevista.
            # Abrir essas imagens ajuda a entender POR QUE o modelo erra
            # (imagem borrada, rótulo errado no dataset, classes parecidas...).

    print(f"\nResultados salvos em {config.DIR_RESULTADOS}")
    # Mensagem final indicando onde estão os arquivos gerados.

if __name__ == "__main__":
    main()
    # Executa main() somente quando o arquivo é rodado diretamente (python evaluate.py).