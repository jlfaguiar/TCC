import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import time
from datetime import datetime
from glob import glob
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
    roc_auc_score, average_precision_score, log_loss, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef
)

sys.path.append(os.path.abspath("C:/Users/jlfag/Desktop/TCC/RPA/api"))
import pipeline_functions

TEMPERATURES = [0.7, 0.9, 0.5]
EXCEL_PATH = "dados_teste_20.xlsx"
TEXT_COLUMN = "Texto"
LABEL_COLUMN = "Sentimento"
BACKUP_INTERVAL_MINUTES = 10

# Carrega os dados
df = pd.read_excel(EXCEL_PATH)
df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

resultados = {}
tempos_conversao = []

start_time = time.time()

for temp in TEMPERATURES:
    print(f"\n🔁 Testando temperatura {temp}...")
    y_true = []
    y_pred = []
    df_detalhado = []
    last_backup = time.time()

    # Restaurar backup mais recente
    backups = sorted(glob(f"backup_temp_{temp}_*.xlsx"), reverse=True)
    start_idx = 0
    if backups:
        print(f"📂 Restaurando backup: {backups[0]}")
        df_backup = pd.read_excel(backups[0])
        df_detalhado = df_backup.values.tolist()
        y_true = df_backup["Sentimento Original"].tolist()
        y_pred = df_backup["Sentimento Previsto"].tolist()
        textos_processados = set(df_backup["Texto Original"].tolist())
        start_idx = df[df[TEXT_COLUMN].isin(textos_processados)].index.max() + 1
        print(f"🔁 Retomando do índice {start_idx}...")
    else:
        print("📂 Nenhum backup encontrado. Iniciando do zero.")

    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        texto = row[TEXT_COLUMN]
        label_real = row[LABEL_COLUMN]

        try:
            start = time.time()
            pred, texto_reescrito = pipeline_functions.classificar_sentimento(texto=texto, temperatura=temp)
            tempos_conversao.append(time.time() - start)
            print(f'Tempo médio para conversão: {sum(tempos_conversao) / len(tempos_conversao):.2f}s')
            label_map = {"Negativo": -1, "Neutro": 0, "Positivo": 1}
            pred_int = label_map.get(pred)
            if pred_int is None:
                print(f"⚠️  Ignorando resposta inválida: {pred}")
                continue

            y_true.append(label_real)
            y_pred.append(pred_int)
            df_detalhado.append([texto, label_real, texto_reescrito, pred_int])

            if time.time() - last_backup > BACKUP_INTERVAL_MINUTES * 60:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_df = pd.DataFrame(df_detalhado, columns=["Texto Original", "Sentimento Original", "Texto Reescrito", "Sentimento Previsto"])
                backup_df.to_excel(f"backup_temp_{temp}_{timestamp}.xlsx", index=False)
                print(f"💾 Backup salvo em backup_temp_{temp}_{timestamp}.xlsx")
                last_backup = time.time()

        except Exception as e:
            print(f"❌ Erro ao processar linha {i}: {e}")

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

        roc = pr_auc = None
        if len(set(y_true)) == 2:
            roc = roc_auc_score(y_true, y_pred)
            pr_auc = average_precision_score(y_true, y_pred)

        resultados[temp] = {
            "accuracy": acc, "f1_score": f1, "precision": prec, "recall": rec,
            "log_loss": logloss, "balanced_accuracy": bal_acc, "cohens_kappa": kappa,
            "mcc": mcc, "roc_auc": roc, "auc_pr": pr_auc, "support": len(y_true),
            "confusion_matrix": conf_mat
        }

        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Matriz de Confusão - Temperatura {temp}")
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_temp_{temp}.png")
        plt.close()

        final_df = pd.DataFrame(df_detalhado, columns=["Texto Original", "Sentimento Original", "Texto Reescrito", "Sentimento Previsto"])
        final_df.to_excel(f"detalhado_temp_{temp}.xlsx", index=False)

        print(f'🌡️ Temperatura {temp} finalizada em {str(time.time() - start_time):.2f}s')
        start_time = time.time()

    except Exception as e:
        print(f"❌ Erro ao calcular métricas para temperatura {temp}: {e}")

# Salva todas as métricas
pd.DataFrame(resultados).T.to_excel("metricas_por_temperatura.xlsx", index_label="Temperatura")
print("\n✅ Avaliação finalizada com backups e restauração automática!")
