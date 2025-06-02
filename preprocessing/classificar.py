import pandas as pd
from transformers import pipeline

# Carregar base limpa
df = pd.read_excel("dados_limpos_ptbr.xlsx")

# Carregar modelo alternativo (multilíngue)
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Converte a classificação para -1, 0 ou +1
def classify_sentiment(text):
    try:
        result = sentiment_model(text[:512])[0]
        label = result["label"]  # Ex: '1 star', '2 stars', ..., '5 stars'
        stars = int(label[0])
        if stars <= 2:
            return -1
        elif stars == 3:
            return 0
        else:
            return 1
    except:
        return 0

# Aplicar a classificação
df["Sentimento"] = df["Texto"].apply(classify_sentiment)

# Exportar resultado
df.to_excel("dados_classificados_sentimento.xlsx", index=False)
print("Arquivo salvo como 'dados_classificados_sentimento.xlsx'")