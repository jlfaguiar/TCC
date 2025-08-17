import torch
import re
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, pipeline
from llama_cpp import Llama

# Carrega modelo e tokenizer
tokenizer = BertTokenizer.from_pretrained("./bertimbau-basic-sentimento")
modelo_classificador = BertForSequenceClassification.from_pretrained("./bertimbau-basic-sentimento")
modelo_embeddings = BertModel.from_pretrained("neuralmind/bert-base-portuguese-cased", output_attentions=True)
llm = Llama(model_path="./llm-models/openhermes.gguf", n_ctx=512, n_threads=4)

# Pipeline HuggingFace para classificação
clf = pipeline(
    "text-classification",
    model=modelo_classificador,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# Extrai tokens importantes pela atenção
def extrair_tokens_importantes(texto: str, top_n: int = 5):
    inputs = tokenizer(texto, return_tensors="pt")
    with torch.no_grad():
        outputs = modelo_embeddings(**inputs)

    attentions = outputs.attentions[-1][0]
    mean_attention = attentions.mean(dim=0).mean(dim=0)

    token_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    tokens_ordenados = sorted(
        zip(tokens, mean_attention.tolist()), key=lambda x: x[1], reverse=True
    )

    return [
        tok for tok, peso in tokens_ordenados
        if tok not in ["[CLS]", "[SEP]", "[PAD]"] and not tok.startswith("##")
    ][:top_n]

# Limpa tags <think> que alguns modelos podem gerar
def remover_bloco_think(texto: str) -> str:
    return re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()

def limpar_reescrita(texto: str) -> str:
    # Remove blocos <think>...</think>
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL)

    # Remove seções explicativas (ex: "---", "### Solution", "Here's one possible...")
    texto = re.sub(r"(?s)(---|\#\#\# Solution:).*", "", texto).strip()
    texto = re.sub(r"^\s*(resposta|comentário reescrito|texto reescrito|texto|comentário)\s*:\s*", "", texto, flags=re.IGNORECASE)
    # Se ainda sobrar explicação, pega só a última linha "frase útil"
    linhas = [linha.strip() for linha in texto.splitlines() if linha.strip()]
    if linhas:
        return linhas[-1]
    return texto.strip()

# LLM auxiliar via LM Studio (localhost)
def reescrever_com_llm(texto_original: str, tokens_relevantes: list[str], temperatura: float = 0.5) -> str:

    prompt_base = """
        Você é um pré-processador textual para um classificador de sentimentos. Sua função é APENAS REESCREVER comentários mantendo o conteúdo original, inclusive termos ofensivos, pois eles são importantes para o modelo que irá classificar o sentimento e o tom.
    
        Reescreva este comentário de forma clara, mantendo o mesmo sentido e tom (inclusive se for ofensivo). Use sinônimos levemente formais ou linguagem levemente mais estruturada. Ele deve ser, em todos os casos, em PORTUGUÊS DO BRASIL.
    
        Comentário original: "{comentario}"
        Tokens importantes: {tokens}
    
        Apenas devolva o texto reescrito, sem censura, sem explicação, sem nenhum comentário adicional.
        """

    tokens_str = ", ".join(tokens_relevantes)
    max_prompt_tokens = llm.n_ctx() - 200

    try:
        # Gera prompt com comentário completo
        prompt = prompt_base.format(comentario=texto_original, tokens=tokens_str)
        prompt_tokens = llm.tokenize(prompt.encode())

        if len(prompt_tokens) > max_prompt_tokens:
            prompt_sem_comentario = prompt_base.format(comentario="", tokens=tokens_str)
            overhead_tokens = len(llm.tokenize(prompt_sem_comentario.encode()))
            limite_comentario = max_prompt_tokens - overhead_tokens

            if limite_comentario > 0:
                comentario_tokens = llm.tokenize(texto_original.encode())
                comentario_cortado = llm.detokenize(comentario_tokens[-limite_comentario:]).decode(errors="ignore")
                prompt = prompt_base.format(comentario=comentario_cortado, tokens=tokens_str)
            else:
                print("[WARN] Prompt base muito longo. Usando versão sem corte.")
    except Exception as e:
        print(f"[ERRO ao ajustar prompt]: {e}")
        prompt = prompt_base.format(comentario=texto_original, tokens=tokens_str)

    print("Tokens no prompt final:", len(llm.tokenize(prompt.encode())))

    resposta = llm.create_completion(
        prompt=prompt,
        temperature=temperatura,
        max_tokens=200,
        seed=1
    )

    conteudo = resposta["choices"][0]["text"]
    return limpar_reescrita(conteudo.strip())


# Função principal de classificação
def classificar_sentimento(texto: str, temperatura: float = 0.5):# -> str
    tokens_relevantes = extrair_tokens_importantes(texto)
    texto_reescrito = reescrever_com_llm(texto, tokens_relevantes, temperatura=temperatura)
    print('Texto reescrito: ' + texto_reescrito)
    pred = clf(texto_reescrito)[0]["label"]

    return {
        "LABEL_0": "Negativo",
        "LABEL_1": "Neutro",
        "LABEL_2": "Positivo"
    }.get(pred, "Desconhecido")


print('Pipeline functions carregado')
