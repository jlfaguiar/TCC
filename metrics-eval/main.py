import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
    roc_auc_score, average_precision_score, log_loss, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import json



# Configurações
API_URL = "http://localhost:8000/classificar/"  # Altere para o endpoint real
TEMPERATURES = [temp/10 for temp in range(3, 16, 4)]     # Temperaturas que serão testadas
EXCEL_PATH = "dados_classificados_sentimento.xlsx"         # Altere para seu arquivo
TEXT_COLUMN = "Texto"
LABEL_COLUMN = "Sentimento"

# Carrega os dados
df = pd.read_excel(EXCEL_PATH)
df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

# Garante que os rótulos sejam inteiros
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

# Função para obter a predição da API
def obter_predicao_api(texto, temperature):
    payload = {"texto": texto, "temperatura": temperature}
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        print('Sucesso')
        return response.json().get("classe")
    except Exception as e:
        print(f"Erro ao consultar API: {e}")
        return None

# Armazena os resultados por temperatura
resultados = {}

# Loop para cada temperatura
for temp in TEMPERATURES:
    print(f"Testando temperatura {temp}...")
    y_true = []
    y_pred = []

    for _, row in df.iterrows():
        texto = row[TEXT_COLUMN]
        label_real = row[LABEL_COLUMN]
        pred = obter_predicao_api(texto, temp)

        if pred is not None:
            y_true.append(label_real)
            y_pred.append(pred)

    # Calcula métricas
    try:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted")
        conf_mat = confusion_matrix(y_true, y_pred)
        logloss = log_loss(y_true, pd.get_dummies(y_pred), labels=list(set(y_true)))
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        # AUC e PR-AUC (somente para binário)
        roc = None
        pr_auc = None
        if len(set(y_true)) == 2:
            roc = roc_auc_score(y_true, y_pred)
            pr_auc = average_precision_score(y_true, y_pred)

        # Salva os resultados
        resultados[temp] = {
            "accuracy": acc,
            "f1_score": f1,
            "precision": prec,
            "recall": rec,
            "log_loss": logloss,
            "balanced_accuracy": bal_acc,
            "cohens_kappa": kappa,
            "mcc": mcc,
            "roc_auc": roc,
            "auc_pr": pr_auc,
            "support": len(y_true),
            "confusion_matrix": conf_mat
        }

        # Exibe matriz de confusão
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Matriz de Confusão - Temperatura {temp}")
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_temp_{temp}.png")
        plt.close()

    except Exception as e:
        print(f"Erro ao calcular métricas para temperatura {temp}: {e}")

# Exibe resultados
df_resultados = pd.DataFrame(resultados).T
print("\nMétricas por Temperatura:")
print(df_resultados)

# Salva em Excel
df_resultados.to_excel("metricas_por_temperatura.xlsx", index_label="Temperatura")
