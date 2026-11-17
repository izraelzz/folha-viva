# Folha Viva

App Flutter para analisar folhas de pepino e melão-amargo. A API usa o modelo
em `modelos_experimental_saude/melhor_modelo.pth` e os metadados em
`data_public/classes.json`; mantenha esses arquivos no projeto.

## Preparação (uma vez)

Na raiz do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
Set-Location mobile
flutter pub get
```

## Iniciar

Abra dois terminais.

Terminal 1 — na raiz do projeto, iniciar a API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Terminal 2 — iniciar o app:

```powershell
Set-Location mobile
flutter run
```

No emulador Android, o app já se conecta à API. Para usar um celular na mesma
rede Wi-Fi, informe o IP do computador:

```powershell
flutter run --dart-define=API_BASE_URL=http://IP-DO-COMPUTADOR:8000
```

Modelo baseado no dataset [Plant Leaf Disease Classification](https://huggingface.co/datasets/Project-AgML/plant_leaf_disease_classification), publicado por Hasan et al. (2024), Mendeley Data, DOI [10.17632/n67gctmjyj.3](https://doi.org/10.17632/n67gctmjyj.3), licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
