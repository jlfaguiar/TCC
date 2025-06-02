import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import torch
import evaluate
from transformers import pipeline

print("CUDA disponível:", torch.cuda.is_available())
print("Dispositivo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nenhuma GPU encontrada")

# 1. Carrega os dados
df = pd.read_excel("dados_treino_80.xlsx")  # ou xlsx convertido pra CSV
df_test = pd.read_excel("dados_teste_20.xlsx")
df = df[["Texto", "Sentimento"]].rename(columns={"Texto": "text", "Sentimento": "label"})
df_test = df_test[["Texto", "Sentimento"]].rename(columns={"Texto": "text", "Sentimento": "label"})
# Reindexar rótulos: -1 → 0, 0 → 1, 1 → 2
df["label"] = df["label"].map({-1: 0, 0: 1, 1: 2})
df_test["label"] = df_test["label"].map({-1: 0, 0: 1, 1: 2})

# 2. Transforma em Dataset Hugging Face
dataset = Dataset.from_pandas(df)
dataset_test = Dataset.from_pandas(df_test)
# 3. Tokenizador e modelo base
tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = BertForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased", num_labels=3)

# 4. Pré-processamento
def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

dataset = dataset.map(preprocess)
dataset_test = dataset_test.map(preprocess)
# 5. Divide treino/teste
dataset_split = dataset.train_test_split(test_size=0.3)

# 5.1 Converte de volta para pandas para salvar
train_df = pd.DataFrame(dataset_split["train"])
test_df = pd.DataFrame(dataset_split["test"])

train_df.to_csv("dados_treino.csv", index=False, encoding='utf-8')
test_df.to_csv("dados_teste.csv", index=False, encoding='utf-8')

# 6. Métricas
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
    }

# 7. Configura o treino
args = TrainingArguments(
    output_dir="./bertimbau-sentimento-base",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# 8. Treina
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    eval_dataset=dataset_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# 9. Pipeline de inferência
clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# 10. Salva modelo e tokenizer
model.save_pretrained("./bertimbau-basic-sentimento")
tokenizer.save_pretrained("./bertimbau-basic-sentimento")

print("Finalizado. Datasets salvos em CSV.")
