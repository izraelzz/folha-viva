import os  # Permite configurar os caminhos do modelo e dos dados por variável de ambiente
import torch  # Importa o PyTorch, a biblioteca de deep learning usada para criar e treinar redes neurais
from pathlib import Path  # Importa a classe Path, que manipula caminhos de arquivos/pastas de forma independente do sistema operacional

# ------------------------------------------------------------
# CAMINHOS DO PROJETO
# ------------------------------------------------------------

RAIZ: Path = Path(__file__).resolve().parent
# __file__ é o caminho deste arquivo .py
# .resolve() transforma em caminho absoluto (completo), resolvendo atalhos como "..".
# .parent pega a pasta onde este arquivo está, que vira a "raiz" do projeto.
# Obs.: __file__ não existe em notebooks Jupyter/Colab; lá isso geraria erro.

DIR_DADOS: Path = Path(os.environ.get("PLANT_DATA_DIR", str(RAIZ / "data")))
# Pasta onde ficam os dados (ex.: imagens de treino e teste).
# O operador "/" com objetos Path junta caminhos: RAIZ/data

DIR_MODELOS: Path = Path(os.environ.get("PLANT_MODEL_DIR", str(RAIZ / "modelos")))
# Pasta onde os modelos treinados (pesos da rede) serão salvos.

DIR_RESULTADOS: Path = Path(os.environ.get("PLANT_RESULTS_DIR", str(RAIZ / "resultados")))
# Pasta para salvar resultados: gráficos, métricas, relatórios etc.

# ------------------------------------------------------------
# HIPERPARÂMETROS DE TREINAMENTO
# ------------------------------------------------------------

TAMANHO_IMAGEM: int = 224
# Tamanho (em pixels) para o qual as imagens serão redimensionadas: 224x224.
# É o tamanho padrão esperado por redes clássicas como ResNet, VGG, etc.

BATCH_SIZE: int = 32
# Quantidade de imagens processadas de uma vez antes de atualizar os pesos.
# Batches maiores usam mais memória da GPU; menores deixam o treino mais "ruidoso".

EPOCAS: int = 10
# Número de épocas: quantas vezes o modelo vai percorrer TODO o conjunto de treino.

LR: float = 1e-3
# Learning rate (taxa de aprendizado) = 0.001.
# Controla o tamanho do "passo" dado ao ajustar os pesos a cada atualização.
# Muito alta: o treino fica instável; muito baixa: o treino fica lento.

SEED: int = 42
# Semente aleatória. Fixá-la permite reproduzir os mesmos resultados
# (mesma inicialização dos pesos, mesma ordem dos dados embaralhados etc.).
# Aqui ela só é definida; precisa ser aplicada em outro ponto do código (ex.: torch.manual_seed(SEED)).

# ------------------------------------------------------------
# NORMALIZAÇÃO DAS IMAGENS
# ------------------------------------------------------------

MEDIA: list[float] = [0.485, 0.456, 0.406]
# Média de cada canal de cor (R, G, B) calculada no dataset ImageNet.
# Usada para normalizar as imagens: pixel_normalizado = (pixel - média) / desvio.

DESVIO: list[float] = [0.229, 0.224, 0.225]
# Desvio padrão de cada canal (R, G, B), também do ImageNet.
# Usar esses valores é importante ao aproveitar modelos pré-treinados no ImageNet
# (transfer learning), pois eles foram treinados com imagens normalizadas assim.

# ------------------------------------------------------------
# ESCOLHA DO HARDWARE (CPU / GPU)
# ------------------------------------------------------------

def dispositivo() -> torch.device:
    # Define uma função que retorna o melhor dispositivo disponível para rodar o modelo.
    # "-> torch.device" é uma anotação de tipo: indica que a função retorna um torch.device.

    if torch.cuda.is_available():
        # Verifica se há uma GPU NVIDIA com CUDA disponível.
        return torch.device("cuda")
        # Se houver, usa a GPU NVIDIA (a opção mais rápida para deep learning).

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # hasattr(...) verifica se esta versão do PyTorch tem suporte a MPS
        # (versões antigas não têm, e acessar direto daria erro).
        # MPS = Metal Performance Shaders: a GPU dos Macs com chip Apple (M1, M2, M3...).
        return torch.device("mps")
        # Se disponível, usa a GPU do Mac.

    return torch.device("cpu")
    # Se nenhuma GPU estiver disponível, usa o processador (CPU).
    # Funciona em qualquer máquina, mas o treino é bem mais lento.