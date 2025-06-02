import pandas as pd
import re
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42

# Carrega o Excel original
df = pd.read_excel("dados_sem_duplicatas.xlsx")

# Define o que é spam ou irrelevante
def is_spam(text):
    if not isinstance(text, str):
        return True
    text = text.strip()
    if len(text) < 5:
        return True
    if re.fullmatch(r'[@#\s\W\d]+', text):
        return True
    return False

# Detecta se o idioma é português
def is_portuguese(text):
    try:
        return detect(text) == 'pt'
    except:
        return False

# Aplica as funções de limpeza
df_clean = df[~df['Texto'].apply(is_spam)]
df_clean = df_clean[df_clean['Texto'].apply(is_portuguese)]

# Exporta para novo Excel
df_clean.to_excel("dados_limpos_ptbr.xlsx", index=False)
print(f"Arquivo salvo como 'dados_limpos_ptbr.xlsx' com {len(df_clean)} registros.")