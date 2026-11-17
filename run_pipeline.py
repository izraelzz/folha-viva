from __future__ import annotations   # permite usar tipos novos (ex.: "str | None") mesmo no Python 3.9

import argparse                      # lê as opções digitadas no terminal (--epocas, --retreinar...)
import ast                           # lê o código Python como estrutura (para ignorar comentários)
import hashlib                       # calcula o "hash" (impressão digital) de um arquivo
import json                          # salva/lê as informações do último treino (treino_info.json)
import os                            # informações do sistema (é Windows ou não?) e conversão de caminhos
import shutil                        # operações com pastas inteiras (ex.: apagar a pasta resultados)
import subprocess                    # executa outros programas (pip, train.py, evaluate.py, api.py)
import sys                           # versão do Python e encerramento do programa (sys.exit)
import venv                          # cria ambientes virtuais (a pasta venv)
from dataclasses import dataclass    # cria classes simples que só guardam dados
from pathlib import Path             # trabalha com caminhos de pastas/arquivos (Windows, Mac e Linux)
from typing import Final, Literal, NoReturn, Optional, Sequence, TypedDict, Union   # ferramentas de type hints


# =====================================================================
# TIPOS
# =====================================================================
# Apelidos de tipos usados no arquivo. Eles não mudam a execução:
# servem para o editor e o mypy entenderem o código e apontarem erros.

# Um pedaço de comando pode ser texto ("train.py"), número (10) ou caminho (Path).
# Ex. de comando completo: [Path(".../python.exe"), "train.py", "--epocas", 10]
# Union[A, B, C] = "pode ser A, B ou C".
# os.PathLike está entre aspas porque só é lido pelo verificador de tipos.
Argumento = Union[str, int, "os.PathLike[str]"]

# Literal = só aceita exatamente estes 3 textos.
# Se alguém escrever "mobilnet" (errado), o editor acusa antes de rodar.
Arquitetura = Literal["mobilenet", "resnet18", "cnn"]

# Tupla com as arquiteturas, usada como opções válidas do --arquitetura.
#   Final                    -> constante: o verificador acusa se alguém tentar mudar o valor
#   tuple[Arquitetura, ...]  -> tupla de qualquer tamanho onde cada item é uma Arquitetura
ARQUITETURAS: Final[tuple[Arquitetura, ...]] = ("mobilenet", "resnet18", "cnn")


# TypedDict = descreve o formato de um dicionário (quais chaves e de que tipo).
# É o conteúdo do arquivo treino_info.json, salvo ao final de cada treino.
class InfoTreino(TypedDict):
    dados: str                 # impressão digital das fotos de data/train e data/val
    codigo: str                # impressão digital do código de treino (config, dataset, modelo, train)
    arquitetura: Arquitetura   # com qual rede o modelo foi treinado
    epocas: int                # com quantas épocas
    batch: int                 # com qual tamanho de lote


# =====================================================================
# CAMINHOS
# =====================================================================
# Todos calculados a partir da pasta deste arquivo (igual ao config.py),
# então funcionam não importa de onde você rode o script.

RAIZ: Final[Path] = Path(__file__).resolve().parent                # pasta do projeto
DIR_VENV: Final[Path] = RAIZ / "venv"                               # onde o ambiente virtual é criado
REQUISITOS: Final[Path] = RAIZ / "requirements.txt"                 # lista de bibliotecas a instalar
MARCA_INSTALACAO: Final[Path] = DIR_VENV / ".requisitos_instalados" # arquivo "lembrete": guarda o hash do
                                                                    # requirements.txt da última instalação
MODELO: Final[Path] = RAIZ / "modelos" / "melhor_modelo.pth"        # modelo treinado (gerado pelo train.py)
RESULTADOS: Final[Path] = RAIZ / "resultados"                       # gráficos e relatórios do treino/avaliação
INFO_TREINO: Final[Path] = RAIZ / "modelos" / "treino_info.json"    # "impressão digital" do último treino completo
TREINO_EM_ANDAMENTO: Final[Path] = RAIZ / "modelos" / ".treino_em_andamento"  # existe só DURANTE o treino;
                                                                    # se sobrar, o treino foi interrompido


# =====================================================================
# O QUE, SE MUDAR, EXIGE TREINAR DE NOVO
# =====================================================================

# Arquivos de código que influenciam o treino.
# (api.py, predict.py e evaluate.py ficam de fora: mudar eles não muda o modelo)
ARQUIVOS_TREINO: Final[tuple[str, ...]] = ("config.py", "dataset.py", "modelo.py", "train.py")

# Pastas de fotos usadas no treino. A data/test fica de fora: ela só serve para avaliar.
PASTAS_DADOS: Final[tuple[str, ...]] = ("train", "val")

# Valores usados quando você não passa a opção no comando (e não há treino anterior)
PADRAO_ARQUITETURA: Final[Arquitetura] = "mobilenet"
PADRAO_EPOCAS: Final[int] = 10
PADRAO_BATCH: Final[int] = 32


# =====================================================================
# OPÇÕES DA LINHA DE COMANDO
# =====================================================================

# @dataclass cria sozinho o __init__ da classe a partir dos campos abaixo.
# Assim dá para fazer Opcoes(arquitetura="cnn", epocas=10, ...) sem escrever o __init__.
# frozen=True = depois de criado, o objeto não pode ser alterado (somente leitura).
@dataclass(frozen=True)
class Opcoes:
    """Opções da linha de comando, já com os tipos certos."""
    # Optional[...] = pode ser None. Fica None quando você NÃO digita a opção no comando;
    # aí o script reaproveita o valor do último treino (ou o padrão, se nunca treinou).
    arquitetura: Optional[Arquitetura]   # --arquitetura: qual rede treinar ("mobilenet", "resnet18" ou "cnn")
    epocas: Optional[int]                # --epocas: quantas épocas de treino
    batch: Optional[int]                 # --batch: quantas fotos por lote
    retreinar: bool                      # --retreinar: True = treina de novo mesmo sem nenhuma mudança
    sem_api: bool                        # --sem-api: True = não sobe a API no final (só treina/avalia)


# =====================================================================
# FUNÇÕES AUXILIARES
# =====================================================================

def titulo(texto: str) -> None:
    """Imprime um cabeçalho destacado para separar as etapas no terminal.

    Ex.: titulo("1/5  Ambiente virtual (venv)") mostra:
        ============================================================
          1/5  Ambiente virtual (venv)
        ============================================================
    """
    print("\n" + "=" * 60)        # pula uma linha e imprime 60 sinais de "="
    print(f"  {texto}")           # o texto, com 2 espaços de recuo
    print("=" * 60, flush=True)   # flush=True: mostra na tela na hora (sem esperar o buffer encher)


def python_da_venv() -> Path:
    """Devolve o caminho do Python que está DENTRO da venv.

    É esse Python que roda o pip, o train.py etc., para usar as bibliotecas
    instaladas na venv (e não as do Python do sistema).
    """
    if os.name == "nt":                                   # "nt" = Windows
        return DIR_VENV / "Scripts" / "python.exe"        # no Windows fica em venv\Scripts\python.exe
    return DIR_VENV / "bin" / "python"                    # no Mac/Linux fica em venv/bin/python


def falhar(mensagem: str, codigo: int = 1) -> NoReturn:
    """Mostra uma mensagem de erro e encerra o programa.

    O tipo de retorno NoReturn avisa ao verificador de tipos que esta função
    NUNCA retorna (ela sempre encerra o programa).
    """
    print(mensagem)
    sys.exit(codigo)   # encerra o programa; código 0 = sucesso, qualquer outro = erro


def rodar(cmd: Sequence[Argumento], descricao: str) -> None:
    """Executa um comando (ex.: pip install, train.py) e para tudo se ele falhar.

    cmd       -> lista com as partes do comando, ex.: [python, "train.py", "--epocas", 10]
    descricao -> texto usado na mensagem de erro, ex.: "treinamento"
    """

    # O subprocess só aceita texto, então converte cada parte do comando:
    #   - se for número (int) -> str(c)        ex.: 10 vira "10"
    #   - senão (texto/Path)  -> os.fspath(c)  ex.: Path("venv/bin/python") vira "venv/bin/python"
    partes: list[str] = [os.fspath(c) if not isinstance(c, int) else str(c) for c in cmd]

    # Mostra o comando que vai rodar (o "$" imita o terminal)
    print(f"$ {' '.join(partes)}", flush=True)

    # Executa o comando e ESPERA ele terminar.
    #   cwd=RAIZ -> roda dentro da pasta do projeto (cwd = "current working directory")
    # A saída do comando (prints, barra do pip etc.) aparece direto no terminal.
    resultado: subprocess.CompletedProcess[bytes] = subprocess.run(partes, cwd=RAIZ)

    # returncode = código de saída do programa: 0 = deu certo; qualquer outro número = deu erro
    if resultado.returncode != 0:
        falhar(f"\n❌ Falhou: {descricao} (código {resultado.returncode}).\n"
               "   Veja a mensagem de erro acima.", resultado.returncode)


# =====================================================================
# ETAPAS DO PIPELINE
# =====================================================================

def criar_venv() -> None:
    """Etapa 1: cria o ambiente virtual, se ainda não existir.

    A venv é uma pasta com um Python "particular" do projeto. As bibliotecas
    instaladas nela não se misturam com as de outros projetos do computador.
    """
    titulo("1/5  Ambiente virtual (venv)")

    # Se o Python da venv já existe, a venv já foi criada antes -> reaproveita e sai da função
    if python_da_venv().exists():
        print(f"✔ venv já existe em {DIR_VENV.name}/ — reaproveitando.")
        return

    print("Criando venv... (alguns segundos)")
    venv.create(DIR_VENV, with_pip=True)   # cria a pasta venv; with_pip=True já instala o pip dentro dela
    print(f"✔ venv criada em {DIR_VENV.name}/")


def instalar_bibliotecas() -> None:
    """Etapa 2: instala o requirements.txt na venv (pula se já foi instalado).

    Para saber se já instalou, compara o hash atual do requirements.txt com o
    hash salvo no arquivo lembrete (MARCA_INSTALACAO) na última instalação.
    """
    titulo("2/5  Instalando bibliotecas")

    # Calcula a "impressão digital" (hash MD5) do requirements.txt:
    #   read_bytes() -> lê o arquivo
    #   md5()        -> gera o hash
    #   hexdigest()  -> transforma em texto, ex.: "3f2a9c..."
    # Se o arquivo mudar (ex.: você adicionou uma biblioteca), o hash muda também.
    hash_atual: str = hashlib.md5(REQUISITOS.read_bytes()).hexdigest()

    # Se o lembrete existe e tem o mesmo hash -> nada mudou desde a última instalação -> pula
    if MARCA_INSTALACAO.exists() and MARCA_INSTALACAO.read_text() == hash_atual:
        print("✔ Bibliotecas já instaladas — pulando.")
        return

    py: Path = python_da_venv()   # usa o pip DA VENV (e não o do Python do sistema)

    # "python -m pip" = roda o pip daquele Python específico
    rodar([py, "-m", "pip", "install", "--upgrade", "pip"], "atualizar o pip")   # atualiza o próprio pip
    print("\nInstalando (o PyTorch é grande, pode levar alguns minutos)...")
    rodar([py, "-m", "pip", "install", "-r", REQUISITOS], "instalar requirements.txt")   # -r = lê a lista do arquivo

    # Só chega aqui se a instalação deu certo (senão o rodar() teria encerrado o programa).
    # Salva o hash no lembrete, para a próxima execução saber que já instalou.
    MARCA_INSTALACAO.write_text(hash_atual)
    print("✔ Bibliotecas instaladas.")


def apagar_modelo_antigo() -> None:
    """Etapa 3: apaga o modelo e os resultados anteriores para treinar do zero."""
    print("\nLimpando treino anterior...")

    if MODELO.exists():
        MODELO.unlink()                    # unlink() = apaga um arquivo
        print(f"✔ Apagado: {MODELO.relative_to(RAIZ)}")   # relative_to: mostra o caminho curto (modelos/...)

    if INFO_TREINO.exists():
        INFO_TREINO.unlink()               # a impressão digital do treino antigo não vale mais

    if RESULTADOS.exists():
        shutil.rmtree(RESULTADOS)          # rmtree() = apaga a pasta inteira, com tudo dentro
        print(f"✔ Apagada: {RESULTADOS.relative_to(RAIZ)}/")

    print("✔ Pronto para treinar do zero.")


def treinar(info: InfoTreino) -> None:
    """Etapa 4: roda o train.py com a arquitetura, épocas e lote definidos em info."""
    titulo(f"4/5  Treinando ({info['arquitetura']}, {info['epocas']} épocas)")

    # Monta o comando, ex.: venv/bin/python train.py --arquitetura mobilenet --epocas 10 --batch 32
    # (os números ficam como int aqui; o rodar() converte tudo para texto)
    cmd: list[Argumento] = [python_da_venv(), "train.py", "--arquitetura", info["arquitetura"],
                            "--epocas", info["epocas"], "--batch", info["batch"]]
    rodar(cmd, "treinamento")


def avaliar() -> None:
    """Etapa 5: roda o evaluate.py para medir a nota no conjunto de teste."""
    titulo("5/5  Avaliando no conjunto de teste")
    rodar([python_da_venv(), "evaluate.py"], "avaliação")


def subir_api() -> None:
    """Sobe a API (api.py) e fica rodando até você apertar Ctrl + C."""
    titulo("Subindo a API  (Ctrl + C para parar)")
    print("Endereço:      http://localhost:8000")
    print("Documentação:  http://localhost:8000/docs")
    print("Endpoint:      POST http://localhost:8000/analisar  (form-data, campo 'imagem')\n")
    try:
        # Usa subprocess.run direto (e não o rodar()) porque a API não "termina":
        # ela fica rodando até ser interrompida, e parar com Ctrl + C não é um erro.
        subprocess.run([os.fspath(python_da_venv()), "api.py"], cwd=RAIZ)
    except KeyboardInterrupt:              # KeyboardInterrupt = você apertou Ctrl + C
        print("\nAPI encerrada.")


# =====================================================================
# DETECÇÃO DE MUDANÇAS (decide se precisa treinar de novo)
# =====================================================================

def hash_dados() -> str:
    """Impressão digital de TODAS as fotos de data/train e data/val.

    Muda se você adicionar, remover, renomear, mover ou editar qualquer foto.
    Usa o CONTEÚDO dos arquivos (e não a data de modificação), então copiar
    o projeto para outra pasta ou outro computador não dispara um novo treino.
    """
    h = hashlib.md5()
    for pasta in PASTAS_DADOS:
        base: Path = RAIZ / "data" / pasta
        # rglob("*") = todos os arquivos dentro da pasta e subpastas.
        # sorted() garante sempre a mesma ordem (senão o hash mudaria à toa).
        for arquivo in sorted(base.rglob("*")):
            if not arquivo.is_file():
                continue                                    # pula as pastas, só interessa arquivo
            # Caminho relativo com "/" (ex.: "train/doente/folha1.jpg"):
            # assim o hash é igual no Windows e no Linux.
            h.update(arquivo.relative_to(RAIZ / "data").as_posix().encode())
            with arquivo.open("rb") as f:                   # "rb" = ler em modo binário (bytes)
                # Lê em pedaços de 1 MB para não carregar fotos enormes inteiras na memória.
                # iter(função, b"") chama f.read() até ele devolver b"" (fim do arquivo).
                for pedaco in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(pedaco)
    return h.hexdigest()


def hash_codigo() -> str:
    """Impressão digital do código de treino, IGNORANDO comentários e docstrings.

    Assim, comentar ou reformatar o código não dispara um novo treino:
    só mudanças que alteram o funcionamento (ex.: trocar o learning rate).
    """
    h = hashlib.md5()
    for nome in ARQUIVOS_TREINO:
        fonte: str = (RAIZ / nome).read_text(encoding="utf-8")
        try:
            # ast.parse transforma o código numa "árvore" (estrutura). Comentários (#)
            # não entram na árvore, e espaços/linhas em branco também não.
            arvore: ast.Module = ast.parse(fonte)
            # Docstrings ("""...""") entram na árvore como texto solto no início de
            # funções/classes/arquivo. Aqui removemos eles também.
            for no in ast.walk(arvore):                     # percorre todos os "nós" da árvore
                if isinstance(no, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    corpo = no.body
                    if (corpo and isinstance(corpo[0], ast.Expr)
                            and isinstance(corpo[0].value, ast.Constant)
                            and isinstance(corpo[0].value.value, str)):
                        no.body = corpo[1:] or [ast.Pass()]  # tira a docstring (deixa "pass" se esvaziar)
            conteudo: str = ast.dump(arvore)                # árvore -> texto, sem comentários
        except SyntaxError:
            conteudo = fonte                                # código com erro: usa o texto puro
        h.update(nome.encode())
        h.update(conteudo.encode())
    return h.hexdigest()


def ler_info_treino() -> InfoTreino | None:
    """Lê o treino_info.json do último treino completo (None se não existir ou estiver corrompido)."""
    if not INFO_TREINO.exists():
        return None
    try:
        info: InfoTreino = json.loads(INFO_TREINO.read_text(encoding="utf-8"))
        return info
    except (json.JSONDecodeError, OSError):
        return None


def salvar_info_treino(info: InfoTreino) -> None:
    """Grava a impressão digital do treino que acabou de terminar."""
    INFO_TREINO.parent.mkdir(exist_ok=True)                 # garante que a pasta modelos/ existe
    INFO_TREINO.write_text(json.dumps(info, indent=2), encoding="utf-8")


def montar_info_atual(opcoes: Opcoes, salvo: InfoTreino | None) -> InfoTreino:
    """Monta a impressão digital da situação ATUAL (fotos, código e opções de agora).

    Para cada opção (arquitetura, épocas, lote):
      1º usa o valor que você digitou no comando;
      2º se não digitou, usa o do último treino;
      3º se nunca treinou, usa o padrão.
    Assim, rodar sem opções nunca dispara um treino só por causa das opções.
    """
    print("Verificando se houve mudanças desde o último treino...", flush=True)
    return InfoTreino(
        dados=hash_dados(),
        codigo=hash_codigo(),
        arquitetura=opcoes.arquitetura or (salvo["arquitetura"] if salvo else PADRAO_ARQUITETURA),
        epocas=opcoes.epocas or (salvo["epocas"] if salvo else PADRAO_EPOCAS),
        batch=opcoes.batch or (salvo["batch"] if salvo else PADRAO_BATCH),
    )


def motivos_para_treinar(opcoes: Opcoes, salvo: InfoTreino | None, atual: InfoTreino) -> list[str]:
    """Compara a situação atual com o último treino e devolve os motivos para treinar.

    Lista vazia = nada mudou = não precisa treinar.
    """
    if opcoes.retreinar:
        return ["você pediu --retreinar"]
    if not MODELO.exists():
        return ["ainda não existe modelo treinado"]
    if TREINO_EM_ANDAMENTO.exists():
        return ["o último treino foi interrompido antes de terminar"]
    if salvo is None:
        return []   # modelo de uma versão antiga do script, sem treino_info.json (tratado no main)

    motivos: list[str] = []
    if atual["dados"] != salvo["dados"]:
        motivos.append("as fotos de data/train ou data/val mudaram")
    if atual["codigo"] != salvo["codigo"]:
        motivos.append(f"o código de treino mudou ({', '.join(ARQUIVOS_TREINO)})")
    # Compara as 3 opções, uma por uma
    for campo in ("arquitetura", "epocas", "batch"):
        if atual[campo] != salvo[campo]:
            motivos.append(f"--{campo} mudou: {salvo[campo]} -> {atual[campo]}")
    return motivos


# =====================================================================
# LEITURA DAS OPÇÕES
# =====================================================================

def ler_opcoes(argv: Sequence[str] | None = None) -> Opcoes:
    """Lê as opções digitadas no terminal e devolve um objeto Opcoes.

    argv -> lista de argumentos. Se for None, usa o que foi digitado no terminal.
            Passar uma lista serve para testar sem terminal,
            ex.: ler_opcoes(["--epocas", "3"])
    """
    p = argparse.ArgumentParser(description="Cria a venv, instala, treina e avalia.")   # description aparece no --help

    # default=None: se você não digitar a opção, ela fica None (e o script reaproveita a do último treino)
    p.add_argument("--arquitetura", default=None, choices=ARQUITETURAS)   # choices: só aceita as 3 opções
    p.add_argument("--epocas", type=int, default=None)                   # type=int: converte "10" em 10
    p.add_argument("--batch", type=int, default=None)

    # action="store_true": opção sem valor (liga/desliga).
    # Se aparecer no comando vira True; se não aparecer, fica False.
    p.add_argument("--retreinar", action="store_true", help="treina de novo mesmo sem nenhuma mudança")
    p.add_argument("--sem-api", action="store_true", help="não sobe a API no final")

    # Lê e valida as opções. Se algo estiver errado (ex.: --epocas abc),
    # o argparse mostra o erro e encerra sozinho.
    ns: argparse.Namespace = p.parse_args(argv)

    # Copia do Namespace (que não tem tipos, tudo é "Any") para a dataclass Opcoes (com tipos).
    # Obs.: o argparse troca "-" por "_" nos nomes: --sem-api vira ns.sem_api
    return Opcoes(
        arquitetura=ns.arquitetura,
        epocas=ns.epocas,
        batch=ns.batch,
        retreinar=ns.retreinar,
        sem_api=ns.sem_api,
    )


# =====================================================================
# PROGRAMA PRINCIPAL
# =====================================================================

def main(argv: Sequence[str] | None = None) -> None:
    """Prepara o ambiente, treina SÓ se algo mudou e sobe a API.

    - 1ª vez (sem modelo):   venv -> instalação -> treino -> avaliação -> API
    - Nada mudou:            pula tudo -> API direto
    - Fotos, código de treino ou opções mudaram: treina de novo -> API
    - Com --retreinar:       treina de novo mesmo sem mudanças -> API
    """
    opcoes: Opcoes = ler_opcoes(argv)

    # --- Verificações antes de começar ---
    if sys.version_info < (3, 9):                      # Python muito antigo
        falhar(f"❌ Precisa de Python 3.9+. Você está usando {sys.version.split()[0]}.")
    if not (RAIZ / "data" / "train").exists():         # sem a pasta de fotos (projeto incompleto)
        falhar("❌ Pasta data/train não encontrada. Rode este script dentro da pasta do projeto.")

    # --- Etapas 1 e 2: sempre rodam (mas pulam sozinhas se já foram feitas antes) ---
    criar_venv()
    instalar_bibliotecas()

    # --- Decide se precisa treinar ---
    titulo("3/5  Verificando se precisa treinar")
    salvo: InfoTreino | None = ler_info_treino()       # como estava no último treino completo
    atual: InfoTreino = montar_info_atual(opcoes, salvo)   # como está agora
    motivos: list[str] = motivos_para_treinar(opcoes, salvo, atual)

    if motivos:
        # --- Algo mudou (ou nunca treinou): treina do zero ---
        print("Vai treinar porque:")
        for motivo in motivos:
            print(f"  • {motivo}")

        apagar_modelo_antigo()

        # Marca "treino em andamento". Se o treino for interrompido (Ctrl + C, erro,
        # computador desligou...), essa marca sobra e a próxima execução treina de novo,
        # em vez de usar um modelo pela metade.
        TREINO_EM_ANDAMENTO.parent.mkdir(exist_ok=True)
        TREINO_EM_ANDAMENTO.touch()                     # touch() = cria um arquivo vazio

        treinar(atual)

        # Só chega aqui se o treino terminou com sucesso (senão o rodar() teria encerrado)
        salvar_info_treino(atual)                       # guarda a impressão digital deste treino
        TREINO_EM_ANDAMENTO.unlink()                    # treino completo: remove a marca

        avaliar()
        titulo("✅ Treino concluído!")
        print(f"Modelo:     {MODELO.relative_to(RAIZ)}")
        print(f"Gráficos:   {RESULTADOS.relative_to(RAIZ)}/")

    elif salvo is None:
        # Existe modelo, mas sem treino_info.json: foi treinado por uma versão antiga
        # deste script. Em vez de treinar de novo, "adota" esse modelo e salva a impressão
        # digital de agora. A partir daí, qualquer mudança passa a ser detectada.
        salvar_info_treino(atual)
        print(f"✔ Usando o modelo existente: {MODELO.relative_to(RAIZ)}")
        print("  (treinado antes deste controle existir; mudanças futuras serão detectadas)")

    else:
        # --- Nada mudou: reaproveita o modelo ---
        print("✔ Nada mudou desde o último treino — pulando o treino.")
        print(f"  Modelo: {MODELO.relative_to(RAIZ)}  "
              f"({atual['arquitetura']}, {atual['epocas']} épocas, lote {atual['batch']})")
        print("  (para forçar um novo treino: python run_pipeline.py --retreinar)")

    # --- Sobe a API (a não ser que tenha usado --sem-api) ---
    if opcoes.sem_api:
        print("\nAPI não iniciada (--sem-api). Para subir: python run_pipeline.py")
    else:
        subir_api()


# =====================================================================
# PONTO DE ENTRADA
# =====================================================================
# Este bloco só roda quando você executa o arquivo direto (python run_pipeline.py).
# Se outro arquivo fizer "import run_pipeline", ele NÃO roda.

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:              # Ctrl + C no meio do processo (ex.: durante o treino)
        print("\n\nInterrompido pelo usuário.")
        sys.exit(1)                        # encerra com código 1 = não terminou normalmente