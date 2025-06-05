# from typing import tuple
import fasttext

FASTTEXT_MODEL_PATH = "/import/ml-sc-scratch3/nidhih/mm_hungarian/downloads/lid.176.bin"

def run_fasttext_langdetect_model(model, text: str) -> tuple[str, float]:
    result = model.predict(text.replace("\n", ""), k=1)

    lang_code = result[0][0].split("__")[-1]
    prob = round(result[1][0], 2)
    return lang_code, prob

def load_fasttext_model():
    return fasttext.load_model(FASTTEXT_MODEL_PATH)
    
def detect_lang_whole_page_fasttext(model, text: str, top_k_lang: int = 2):
    result = model.predict(text.replace("\n", ""), k=top_k_lang)

    lang_code = result[0][0].split("__")[-1]
    # lang_codes = {result[0][i]: result[-1][i] for i in range(len(result[0]))}
    lang_codes = {result[0][i]: True for i in range(len(result[0]))}

    return lang_codes
    # prob = round(result[1][0], 2)
    # return lang_code, prob