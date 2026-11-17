import io
import json
import os
# io: permite tratar dados em memória (bytes) como se fossem um arquivo.

import time
# time: mede o tempo da análise.

from collections.abc import Callable
# Callable: tipo para "algo que pode ser chamado como função".

from pathlib import Path
# Manipula caminhos de arquivos.

import torch
# PyTorch.

import uvicorn
# Uvicorn: servidor web que executa a aplicação FastAPI e responde às requisições HTTP.

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
# FastAPI: framework para criar APIs web em Python.
#   FastAPI       -> a classe da aplicação
#   File          -> indica que um parâmetro vem como arquivo enviado no formulário
#   HTTPException -> gera respostas de erro HTTP (400, 413...)
#   UploadFile    -> representa o arquivo recebido
# Obs.: para receber arquivos, o FastAPI exige o pacote "python-multipart" instalado.

from PIL import Image, UnidentifiedImageError
# Image: abre e manipula imagens.
# UnidentifiedImageError: erro gerado quando o arquivo não é uma imagem reconhecível.

from pydantic import BaseModel
# BaseModel: define o formato (esquema) dos dados das respostas.
# O FastAPI usa isso para validar os dados e gerar a documentação automática.

from fastapi.middleware.cors import CORSMiddleware

import config
# Caminhos e escolha do dispositivo.

from dataset import transform_avaliacao
# Mesmas transformações usadas na validação/teste.

from modelo import carregar_checkpoint
# Carrega o modelo treinado.

# ------------------------------------------------------------
# CONSTANTES
# ------------------------------------------------------------

if "PLANT_MODEL_PATH" in os.environ:
    CAMINHO_MODELO = Path(os.environ["PLANT_MODEL_PATH"])
elif "PLANT_MODEL_DIR" in os.environ:
    CAMINHO_MODELO = config.DIR_MODELOS / "melhor_modelo.pth"
else:
    CAMINHO_MODELO = (
        config.RAIZ / "modelos_experimental_saude" / "melhor_modelo.pth"
    )
if "PLANT_DATA_DIR" in os.environ or (
    "PLANT_MODEL_DIR" in os.environ and "PLANT_MODEL_PATH" not in os.environ
):
    CAMINHO_CLASSES = config.DIR_DADOS / "classes.json"
else:
    CAMINHO_CLASSES = config.RAIZ / "data_public" / "classes.json"
# Arquivo do modelo que a API vai usar.

_CULTURAS_PERMITIDAS = {"pepino", "melão-amargo"}
_ORIENTACOES_DOENCAS = {
    "cucumber__anthracnose_lesions": (
        "Se essa suspeita for confirmada, remova folhas muito afetadas, descarte "
        "restos doentes e evite molhar ou manusear a folhagem quando estiver úmida."
    ),
    "cucumber__downy_mildew": (
        "Se essa suspeita for confirmada, evite molhar as folhas, melhore a "
        "circulação de ar e retire folhas muito afetadas sem espalhar os restos."
    ),
    "bitter_gourd__downey_mildew": (
        "Se essa suspeita for confirmada, evite molhar as folhas, melhore a "
        "circulação de ar e retire folhas muito afetadas sem espalhar os restos."
    ),
    "bitter_gourd__fusarium_wilt": (
        "A murcha de Fusarium não tem cura caseira. Se confirmada, remova a "
        "planta afetada com cuidado, limpe as ferramentas e peça orientação "
        "antes de replantar no mesmo local."
    ),
    "bitter_gourd__mosaic_virus": (
        "Viroses não têm cura por remédio. Se a suspeita for confirmada, remova "
        "a planta afetada, limpe as ferramentas e controle plantas espontâneas "
        "próximas que possam manter a virose."
    ),
}

TAMANHO_MAX_MB: int = 10
# Tamanho máximo de imagem aceito, em megabytes.

TAMANHO_MAX_BYTES: int = TAMANHO_MAX_MB * 1024 * 1024

CONFIANCA_MINIMA: float = 0.70
# Corte preliminar para não apresentar previsões fracas como um diagnóstico.
# O mesmo limite convertido em bytes (1 MB = 1024 × 1024 bytes).

# ------------------------------------------------------------
# CRIAÇÃO DA APLICAÇÃO
# ------------------------------------------------------------

app: FastAPI = FastAPI(
    title="Detector de Saúde de Plantas",
    description="Envie uma foto de folha para estimar a condição visual da planta.",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
# Cria a aplicação web. Título, descrição e versão aparecem na documentação
# automática, acessível em http://localhost:8000/docs.

# ------------------------------------------------------------
# CARREGAMENTO DO MODELO (UMA ÚNICA VEZ)
# ------------------------------------------------------------

device: torch.device = config.dispositivo()
# Escolhe GPU NVIDIA, GPU do Mac ou CPU.

modelo: torch.nn.Module
classes: list[str]
# Declaração dos tipos.

modelo, classes = carregar_checkpoint(CAMINHO_MODELO, device)
# Carrega o modelo quando a API inicia, e não a cada requisição.
# Carregar um modelo é lento; fazendo isso uma vez só, cada análise fica rápida.

transform: Callable[[Image.Image], torch.Tensor] = transform_avaliacao()
# Cria as transformações uma vez e reaproveita em todas as requisições.
# O tipo diz: "função que recebe uma imagem PIL e devolve um tensor".

METADADOS_CLASSES: dict[str, dict[str, str | bool]] = (
    json.loads(CAMINHO_CLASSES.read_text(encoding="utf-8"))
    if CAMINHO_CLASSES.exists()
    else {}
)
# Metadados opcionais gerados pelo preparador do dataset multiclasse.

# ------------------------------------------------------------
# FORMATO DAS RESPOSTAS (ESQUEMAS)
# ------------------------------------------------------------

class StatusResposta(BaseModel):
    # Formato da resposta da rota "/" (verificar se a API está no ar).

    status: str
    # Ex.: "online".

    modelo: str
    # Nome do arquivo do modelo carregado.

    classes: list[str]
    # Classes que o modelo reconhece.

    dispositivo: str
    # Onde o modelo está rodando (cpu, cuda, mps).

    uso: str
    # Instrução rápida de como usar a API.

class AlternativaResposta(BaseModel):
    classe: str
    # Nome amigável da cultura e condição prevista.

    planta: str
    # Cultura prevista.

    condicao: str
    # Condição prevista.

    confianca: float
    # Pontuação softmax do modelo (não é probabilidade clínica calibrada).

    saudavel: bool
    # Indica se a classe representa uma folha saudável.

class AnaliseResposta(BaseModel):
    # Formato da resposta da rota "/analisar".

    arquivo: str | None
    # Nome do arquivo enviado (pode ser None se o cliente não informar).

    resultado: str
    # Classe prevista (ex.: "doente").

    confianca: float
    # Pontuação softmax da classe prevista (0 a 1).

    probabilidades: dict[str, float]
    # Pontuação softmax de todas as classes.

    mensagem: str
    # Texto explicativo para o usuário.

    tempo_ms: float
    # Tempo gasto na análise, em milissegundos.

    alternativas: list["AlternativaResposta"]
    # As três classes mais prováveis para ajudar a interpretar o resultado.

    orientacao: str
    # Próximo passo seguro, sem prescrever produto fitossanitário.

    incerta: bool
    # True quando a pontuação máxima fica abaixo do corte experimental.

    cultura_incompativel: bool = False
    # True quando o modelo favorece outra cultura que a informada pelo usuário.

def nome_classe(classe: str) -> str:
    dados = METADADOS_CLASSES.get(classe)
    if dados is None:
        return classe.replace("_", " ").capitalize()
    return f"{dados['planta']} — {dados['condicao']}"


def classe_e_saudavel(classe: str) -> bool:
    dados = METADADOS_CLASSES.get(classe)
    if dados is not None:
        return bool(dados["saudavel"])
    return classe == "saudavel"


def indices_da_planta(planta: str | None) -> tuple[list[int], str | None]:
    if planta is None:
        raise HTTPException(
            400,
            "Selecione uma cultura: Pepino ou Melão-amargo.",
        )

    solicitada = planta.strip().casefold()
    if solicitada not in _CULTURAS_PERMITIDAS:
        raise HTTPException(
            400,
            "Cultura não disponível. Selecione Pepino ou Melão-amargo.",
        )

    indices: list[int] = []
    nome_canonico: str | None = None
    for indice, classe in enumerate(classes):
        nome_planta = METADADOS_CLASSES.get(classe, {}).get("planta")
        if isinstance(nome_planta, str) and nome_planta.casefold() == solicitada:
            indices.append(indice)
            nome_canonico = nome_planta

    if not indices:
        raise HTTPException(
            400,
            "A cultura selecionada não está disponível neste modelo multiclasse.",
        )

    return indices, nome_canonico

# ------------------------------------------------------------
# ROTA GET "/" : STATUS DA API
# ------------------------------------------------------------

@app.get("/")
# Decorador: diz ao FastAPI que a função abaixo responde a requisições GET na rota "/".
# GET é o método usado para CONSULTAR informações (é o que o navegador faz ao abrir um link).

def status() -> StatusResposta:
    return StatusResposta(
        status="online",
        modelo=CAMINHO_MODELO.name,
        # .name pega só o nome do arquivo: "melhor_modelo.pth".

        classes=[nome_classe(classe) for classe in classes],
        dispositivo=str(device),
        # Converte o device em texto (ex.: "cuda").

        uso="POST /analisar com form-data, campo 'imagem'",
    )
    # O FastAPI converte esse objeto automaticamente em JSON na resposta.

# ------------------------------------------------------------
# ROTA POST "/analisar" : CLASSIFICAR UMA IMAGEM
# ------------------------------------------------------------

@app.post("/analisar")
# Esta função responde a requisições POST na rota "/analisar".
# POST é o método usado para ENVIAR dados ao servidor (aqui, a imagem).

async def analisar(
    imagem: UploadFile = File(..., description="Foto da folha (jpg, png...)"),
    # O arquivo chega num campo de formulário chamado "imagem".
    # File(...): os três pontos (Ellipsis) indicam que o campo é OBRIGATÓRIO.
    # "async def" permite usar "await" para ler o arquivo sem travar o servidor.

    planta: str | None = Form(
        default=None,
        description="Cultura conhecida da folha; quando informada, restringe as classes.",
    ),

) -> AnaliseResposta:

    conteudo: bytes = await imagem.read()
    # Lê todo o conteúdo do arquivo enviado como bytes.
    # "await" espera a leitura terminar.

    if not conteudo:
        raise HTTPException(400, "Arquivo vazio.")
        # Se o arquivo veio vazio, devolve erro 400 (Bad Request = requisição inválida).

    if len(conteudo) > TAMANHO_MAX_BYTES:
        raise HTTPException(413, f"Imagem maior que {TAMANHO_MAX_MB} MB.")
        # Se passar do limite, devolve erro 413 (Payload Too Large = conteúdo grande demais).

    try:
        img: Image.Image = Image.open(io.BytesIO(conteudo)).convert("RGB")
        # io.BytesIO transforma os bytes em um "arquivo em memória",
        # permitindo que o PIL abra a imagem sem salvá-la em disco.
        # .convert("RGB") garante 3 canais.

    except UnidentifiedImageError:
        raise HTTPException(400, "O arquivo enviado não é uma imagem válida.") from None
        # Se o arquivo não for imagem (ex.: um PDF), devolve erro 400 com mensagem clara.
        # "from None" esconde o erro original do PIL, deixando o registro de erros mais limpo.

    t0: float = time.perf_counter()
    # Marca o início da medição. perf_counter é mais preciso que time.time()
    # para medir intervalos curtos.

    x: torch.Tensor = transform(img).unsqueeze(0).to(device)
    # Aplica as transformações -> (3, 224, 224);
    # unsqueeze(0) cria um lote de uma imagem -> (1, 3, 224, 224);
    # .to(device) envia para o dispositivo do modelo.

    with torch.no_grad():
        probs: torch.Tensor = torch.softmax(modelo(x), dim=1)[0].cpu()

    classes_da_planta, planta_selecionada = indices_da_planta(planta)
    probabilidades_por_planta: dict[str, float] = {}
    for nome, probabilidade in zip(classes, probs.tolist()):
        nome_planta = METADADOS_CLASSES.get(nome, {}).get("planta")
        if isinstance(nome_planta, str):
            probabilidades_por_planta[nome_planta] = (
                probabilidades_por_planta.get(nome_planta, 0.0) + probabilidade
            )

    planta_prevista = (
        max(probabilidades_por_planta, key=probabilidades_por_planta.get)
        if probabilidades_por_planta
        else None
    )
    cultura_incompativel = (
        planta_selecionada is not None and planta_prevista != planta_selecionada
    )
    indices_resultado = classes_da_planta
    idx: int = max(indices_resultado, key=lambda indice: float(probs[indice]))
    classe: str = classes[idx]
    incerta: bool = cultura_incompativel or float(probs[idx]) < CONFIANCA_MINIMA
    indices_melhores = (
        []
        if cultura_incompativel
        else sorted(
            indices_resultado,
            key=lambda indice: float(probs[indice]),
            reverse=True,
        )[:3]
    )
    alternativas: list[AlternativaResposta] = [
        AlternativaResposta(
            classe=nome_classe(classes[indice]),
            planta=str(METADADOS_CLASSES.get(classes[indice], {}).get("planta", "")),
            condicao=str(
                METADADOS_CLASSES.get(classes[indice], {}).get(
                    "condicao", nome_classe(classes[indice])
                )
            ),
            confianca=round(float(probs[indice]), 4),
            saudavel=classe_e_saudavel(classes[indice]),
        )
        for indice in indices_melhores
    ]

    if cultura_incompativel:
        mensagem = (
            f"Não foi possível analisar esta imagem como {planta_selecionada}. "
            "Nenhuma doença foi indicada. Confira a cultura e tente uma foto nítida "
            "de uma folha."
        )
        orientacao = (
            "Confira se a cultura é pepino ou melão-amargo e selecione a correta. "
            "Nenhuma doença foi indicada."
        )
    elif incerta:
        mensagem = (
            "Não foi possível identificar a condição com segurança. "
            "Tente outra foto com a folha inteira, em foco e bem iluminada."
        )
        if planta_selecionada is not None:
            mensagem = f"Considerando {planta_selecionada}: {mensagem}"
    elif METADADOS_CLASSES.get(classe) is not None:
        mensagem = (
            f"Considerando {planta_selecionada + ', ' if planta_selecionada else ''}"
            f"a imagem tem padrões compatíveis com "
            f"{METADADOS_CLASSES[classe]['condicao']} em {METADADOS_CLASSES[classe]['planta']}."
        )
    else:
        mensagem = (
            "A folha parece saudável."
            if classe == "saudavel"
            else "A folha apresenta sinais que merecem inspeção."
        )

    if not cultura_incompativel:
        if incerta:
            orientacao = (
                "O modelo está incerto. Fotografe outra folha em boa luz e procure assistência "
                "agronômica se os sinais persistirem."
            )
        elif classe_e_saudavel(classe):
            orientacao = (
                "Continue monitorando a planta e mantenha os cuidados adequados à cultura."
            )
        else:
            orientacao = _ORIENTACOES_DOENCAS.get(
                classe,
                "Esta suspeita visual precisa ser confirmada. Registre outras folhas "
                "e procure orientação agronômica antes de agir.",
            )

    return AnaliseResposta(
        arquivo=imagem.filename,
        # Nome original do arquivo enviado.

        resultado=(
            "incerto"
            if cultura_incompativel
            else nome_classe(classe) if classe in METADADOS_CLASSES else classe
        ),

        confianca=0.0 if cultura_incompativel else round(float(probs[idx]), 4),
        # Probabilidade da classe prevista, arredondada para 4 casas.

        probabilidades=(
            {}
            if cultura_incompativel
            else {
                nome_classe(classes[indice]): round(float(probs[indice]), 4)
                for indice in indices_resultado
            }
        ),
        # Dict comprehension: junta cada nome de classe à sua probabilidade.

        mensagem=mensagem,
        # .get devolve a mensagem da classe ou "" se ela não estiver no dicionário
        # (evita erro caso exista uma classe sem mensagem cadastrada).

        tempo_ms=round((time.perf_counter() - t0) * 1000, 1),
        # Tempo decorrido em milissegundos (segundos × 1000), com 1 casa decimal.
        # Mede só a transformação + previsão, não a leitura/abertura da imagem.
        alternativas=alternativas,
        orientacao=orientacao,
        incerta=incerta,
        cultura_incompativel=cultura_incompativel,
    )
    # O FastAPI converte esse objeto em JSON e envia ao cliente.

# ------------------------------------------------------------
# INICIAR O SERVIDOR
# ------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000)
    # Inicia o servidor quando o arquivo é executado diretamente (python api.py).
    # "api:app" significa: no arquivo api.py, use a variável app.
    #   (por isso o arquivo precisa se chamar api.py)
    # host="0.0.0.0": aceita conexões de outras máquinas da rede, não só do próprio computador.
    #   Para aceitar apenas conexões locais, usaria "127.0.0.1".
    # port=8000: a API fica disponível em http://localhost:8000