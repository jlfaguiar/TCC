import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
    roc_auc_score, average_precision_score, log_loss, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o arquivo enviado
df = pd.read_excel("detalhado_temp_0.3.xlsx")

# Preparar os valores reais e previstos
y_true = df["Sentimento Original"]
y_pred = df["Sentimento Previsto"]

# Calcular métricas
results = {}
results["accuracy"] = accuracy_score(y_true, y_pred)
results["f1_score"] = f1_score(y_true, y_pred, average="weighted")
results["precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
results["recall"] = recall_score(y_true, y_pred, average="weighted")
results["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
results["log_loss"] = log_loss(y_true, pd.get_dummies(y_pred), labels=list(set(y_true)))
results["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
results["cohens_kappa"] = cohen_kappa_score(y_true, y_pred)
results["matthews_corrcoef"] = matthews_corrcoef(y_true, y_pred)
results["support"] = len(y_true)

# Verifica se é binário para calcular ROC AUC e AUC PR
if len(set(y_true)) == 2:
    results["roc_auc"] = roc_auc_score(y_true, y_pred)
    results["auc_pr"] = average_precision_score(y_true, y_pred)
else:
    results["roc_auc"] = None
    results["auc_pr"] = None

# Criar DataFrame com as métricas
metricas_df = pd.DataFrame.from_dict(results, orient='index', columns=["Valor"])

# Salvar as métricas no Excel
metrics_output_path = "metricas_temp_0.3.xlsx"
metricas_df.to_excel(metrics_output_path)

# Plotar e salvar a matriz de confusão
plt.figure(figsize=(6, 4))
sns.heatmap(results["confusion_matrix"], annot=True, fmt='d', cmap='Blues',
            xticklabels=["Negativo", "Neutro", "Positivo"],
            yticklabels=["Negativo", "Neutro", "Positivo"])
plt.title("Matriz de Confusão - Temperatura 0.3")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.tight_layout()

conf_matrix_output_path = "matriz_confusao_temp_0.3.png"
plt.savefig(conf_matrix_output_path)
plt.close()

