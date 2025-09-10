# ML_dataset_iris_mlflow_commented.py
# Script educativo: treina um RandomForest no dataset Iris e registra tudo no MLflow.
# Execute: python ML_dataset_iris_mlflow_commented.py

# -------------------------
# 1) Imports
# -------------------------
import os                              # manipular variáveis/paths
import mlflow                           # cliente principal do MLflow
import mlflow.sklearn                   # helpers para modelos scikit-learn
from mlflow.models.signature import infer_signature  # pra gerar signature do modelo
import pandas as pd                     # manipulação tabular
import numpy as np                      # arrays / utilitários numéricos
from sklearn.datasets import load_iris  # dataset exemplo
from sklearn.model_selection import train_test_split  # dividir treino/teste
from sklearn.ensemble import RandomForestClassifier   # modelo de exemplo
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt         # pra salvar figuras (matriz de confusão)
import seaborn as sns                   # visualização (heatmap da matriz)

# -------------------------
# 2) Tracking URI (escolha uma opção)
# -------------------------
# IMPORTANTE: escolha *apenas uma* das opções abaixo, descomentando-a conforme seu fluxo.

# Opção A: se você já abriu `mlflow ui` em outro terminal (modo servidor HTTP), use:
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Opção B: se você prefere gravar só localmente (sem servidor HTTP), use caminho file:///...
# Exemplo Windows (substitua pelo seu diretório de trabalho):
# mlflow.set_tracking_uri("file:///C:/Users/Igor/Documents/Programacao/ws-vscode/mlruns")

# NOTA: Não descomente as duas — escolha a que combina com o que você está rodando.

# -------------------------
# 3) Experimento
# -------------------------
mlflow.set_experiment("iris_random_forest")
# -> cria (ou seleciona) um experimento chamado "iris_random_forest" no tracking server/local.

# -------------------------
# 4) Carregar e preparar dados
# -------------------------
iris = load_iris()                            # carrega o dataset Iris
X = pd.DataFrame(iris.data, columns=iris.feature_names)  # transforma em DataFrame (mais legível)
y = pd.Series(iris.target, name="target")     # target como Series (nome "target")

# -------------------------
# 5) Separar treino/teste
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# stratify=y -> mantém proporção de classes entre treino/teste (boa prática)

# -------------------------
# 6) Hiperparâmetros do modelo
# -------------------------
n_estimators = 100
max_depth = 5
random_state = 42

# -------------------------
# 7) Abrir run e registrar (uso do 'with' fecha o run automaticamente)
# -------------------------
with mlflow.start_run(run_name=f"rf_ne{n_estimators}_md{max_depth}"):
    # ---- parâmetros ----
    mlflow.log_param("n_estimators", n_estimators)  # registra parâmetro
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)

    # ---- treinar modelo ----
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)  # treino do modelo

    # ---- previsões e métricas ----
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)  # acurácia
    # relatório de classificação (string com precisão/recall/f1 por classe)
    clf_report = classification_report(y_test, y_pred, target_names=iris.target_names)

    # registra métricas simples
    mlflow.log_metric("accuracy", float(acc))

    # registra relatório de classificação como artefato (arquivo .txt)
    with open("classification_report.txt", "w") as f:
        f.write(clf_report)
    mlflow.log_artifact("classification_report.txt")  # faz upload do arquivo para os artefatos do run

    # ---- matriz de confusão (salva como imagem e registra artefato) ----
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png")    # salva localmente
    mlflow.log_artifact("confusion_matrix.png")  # registra como artefato no run
    plt.close(fig)  # libera memória da figura

    # ---- feature importances (salva CSV e registra) ----
    fi = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    fi.to_csv("feature_importances.csv", index=False)
    mlflow.log_artifact("feature_importances.csv")

    # ---- assinatura e exemplo de entrada (ajuda reprodutibilidade) ----
    # infer_signature pega uma amostra de entrada/saída para descrever o contrato do modelo
    signature = infer_signature(X_train, model.predict(X_train))
    # input_example: um pequeno DataFrame exemplificando o formato de entrada
    input_example = X_train.head(5)

    # log do modelo (modelo + signature + input example são armazenados nos artefatos)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model_random_forest",  # pasta dentro dos artefatos do run
        signature=signature,
        input_example=input_example
    )
    # Observação: às vezes o MLflow emite avisos sobre 'artifact_path' ou 'signature' — são avisos, o log normalmente funciona.

    # ---- informação final do run ----
    run_id = mlflow.active_run().info.run_id
    print(f"Run registrado: {run_id} — acurácia: {acc:.4f}")
# fim do with -> mlflow.end_run() é chamado automaticamente aqui


'''
Explicações extras (pontos importantes)

Imports

mlflow/mlflow.sklearn: functions para logar parâmetros, métricas, artefatos e para logar modelos sklearn.

infer_signature: ajuda a salvar signature (tipo/formato de entrada e saída), útil quando você colocar o modelo em produção.

Tracking URI — escolha entre 2 modos

Servidor HTTP (mlflow ui rodando): bom se você quer ver o UI em tempo real e usar o Model Registry. Use mlflow.set_tracking_uri("http://127.0.0.1:5000") antes de iniciar o run.
-> Lembre: se configurar HTTP, certifique-se de ter subido mlflow ui num terminal (mlflow ui) antes ou simultaneamente.

Modo local (file): se preferir não rodar server, grave tudo em mlruns/ com file:///caminho/para/mlruns. Para visualizar depois, rode mlflow ui --backend-store-uri file:///C:/.../mlruns.

with mlflow.start_run()

O with é recomendado: garante que o run será finalizado (chamada automática a mlflow.end_run()), evita runs "pendurados" no estado RUNNING.

Parâmetros vs Métricas vs Artefatos vs Modelos

mlflow.log_param() -> registros de configuração (hiperparâmetros).

mlflow.log_metric() -> números/medidas (a cada run).

mlflow.log_artifact() -> arquivos (plots, relatórios, CSVs, etc.).

mlflow.sklearn.log_model() -> empacota o modelo (pickle + metadata) nos artefatos do run; opcionalmente registra assinatura e um exemplo de entrada.

Signature e input_example

signature = infer_signature(X_train, model.predict(X_train)) cria um esquema de entrada/saída.

input_example é útil para auto-documentar como chamar o modelo (ajuda o deploy/serving e a reprodutibilidade).

Aviso/Warning que você viu (Model logged without a signature and input example) desaparece quando você fornece ambos.

Onde os arquivos ficam?

Estrutura típica (modo local): ./mlruns/<experiment_id>/<run_id>/artifacts/... — abra essa pasta e verá seus artefatos (imagens, CSV, modelo).

Como rodar (resumo prático)

Em um terminal: (opção A — servidor)

rode o UI: mlflow ui

em outro terminal rode: python ML_dataset_iris_mlflow_commented.py

abra http://127.0.0.1:5000

Em modo local (sem servidor):

ajuste mlflow.set_tracking_uri("file:///C:/caminho/para/mlruns") no script

rode: python ML_dataset_iris_mlflow_commented.py

rode: mlflow ui --backend-store-uri file:///C:/caminho/para/mlruns

abra http://127.0.0.1:5000

Boas práticas rápidas

Sempre use with mlflow.start_run() (fecha automaticamente).

Logue versão do código (ex.: mlflow.set_tag("git_commit", "<hash>")) para rastreabilidade do código.

Nomeie experimentos e runs de forma consistente (ex.: set_experiment("nome_projeto"), start_run(run_name=...)).

Salve gráficos e artefatos importantes (matriz de confusão, feature importances, datasets de validação).

Se aparecer erro sobre "tracking URI must be http or https"

Significa que o server espera usar um servidor HTTP para artefatos; solução: use mlflow.set_tracking_uri("http://127.0.0.1:5000") enquanto o mlflow ui estiver rodando. Ou use file:///... em ambos (script e mlflow ui --backend-store-uri).
'''