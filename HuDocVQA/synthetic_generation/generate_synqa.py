import os
import re
import sys
import math
import argparse
import numpy as np
import jsonlines
import pytesseract
import logging
from openai import OpenAI
from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
from tenacity import RetryError, retry, stop_after_attempt, wait_random_exponential, before_sleep_log

os.environ['TESSDATA_PREFIX'] = './tessdata/'

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42
MODEL = 'Meta-Llama-3.3-70B-Instruct'

'''
You are a professional question and answer generation assistant. Given a sample text following "Text: ", please write a question-answer pair that addresses a specific point in the text. Make sure that the generated question has an answer in the text, and use proper nouns if available. Format your answer as "Question: <question>" and "Answer: <answer>", separated by a newline. If there are multiple "Text: " prompts, ensure your answer only references the most recent one. If there is no meaningful question-answer pair that can be generated from the text, then reject it.
'''
HUNGARIAN_SYSTEM_PROMPT = 'Professzionális kérdés-válasz generáló asszisztens vagy. Adott egy minta szöveget a "Szöveg" után, írjon egy kérdés-felelet párost, amely a szöveg egy adott pontjára vonatkozik. Győződjön meg arról, hogy a generált kérdésnek van válasza a szövegben, és ha lehetséges, használjon tulajdonneveket. Formázza válaszát "Kérdés: <kérdés>" és "Válasz: <válasz>" formában, újsorral elválasztva. Ha több „Szöveg:” üzenet is van, győződjön meg arról, hogy a válasz csak a legutóbbira hivatkozik. Ha nincs értelmes kérdés-felelet pár, ami a szövegből generálható, akkor utasítsuk el.'

HUNGARIAN_TEXT_EXAMPLE_0 = 'Elektronika 1. zárthelyi /Elméleti kérdések – A csoport  \n \n \n \n \n NEPTUN KÓD Aláírás nélkül érvénytelen! Minden kérdés 2 pontot ér. A rendelkezésre álló idő 30perc. Csak ezt a lapot lehet \nbeadni, szükség esetén a túloldalra írhat! Beadáskor ezt a lapot hosszában hajtsa össze úgy, hogy a NEPTUN \nkód kívülre kerüljön! \nA feladatokat önállóan, meg nem engedett segítség \nigénybevétele nélkül oldottam meg.  \nNÉV: \n \n \nALÁÍRÁS: 1. \nRajzoljon fel és méretezzen egy RC tagot, amelynek \nidőállandója \n1ms! \n(csak megfelelőenméretezett kapcsolást fogadunk el, az elvi kapcsolási rajz \nönmagában nem ér pontot.) 3. \nMi a fő különbség egy vezető és egy szigetelő anyag \nsávszerkezete között?  5. \nEgy normál aktív tartományban működő bipoláris \ntranzisztor emitterárama 1,005mA, kollektorárama \n1mA. Határozza meg földelt emitteres áramerősítési \ntényezőt! 2. \nRöviden írja le az optocsatoló működését!  4. \nEgy \nideálisnak \ntekinthető \nfeszültségerősítő Egy \nideálisnak \ntekinthető \nfeszültségerősítő bemenetén \n1mV, \nkimenetén \n10V \nfeszültség ,mérhető. Adja meg az erősítést, dB-ben! 6. \nAdja meg egy telítésben működő MOS tranzisztor \nkarakterisztika egyenletét!   7. \nRajzoljon fel egy 8× feszültségerősítésű, műveleti erősítővel megvalósított, és megfelelően méretezett \nfázist nem fordító alapkapcsolást! (csak megfelelően \nméretezett kapcsolást fogadunk el, az elvi kapcsolási \nrajz önmagában nem ér pontot.)   9. \nRajzolja fel közelítően egy npn bipoláris tranzisztor \nföldelt emitteres kimeneti karakterisztikáját! (pontosan \ntüntesse fel a tengelyeken ábrázolt mennyiségek és az \nesetleges paraméter nevét!) 8. Ábrázolja közel léptékhelyesen egy 13V letörési feszültségű szilíciumdióda karakterisztikáját! 10. Rajzolja fel egy normál aktív tartományban működő npn bipoláris tranzisztor földelt emitteres kisjelű \nhelyettesítő képét, és számítsa ki a helyettesítő kép \nelemeit! (A tranzisztor munkaponti emitterárama \n5mA, a földelt emitteres áramerősítési tényező 200, a \ntermikus feszültség 26mV)\n\nKérdés: Mekkora a maximális feszültség, amelyen egy ideális feszültségerősítő mérhető?'
HUNGARIAN_QUESTION_EXAMPLE_0 = 'Mekkora a maximális feszültség, amelyen egy ideális feszültségerősítő mérhető?'
HUNGARIAN_ANSWER_EXAMPLE_0 = 'Egy ideálisnak tekinthető feszültségerősítő bemenetén 1mV, kimenetén 10V feszültség , mérhető.'

HUNGARIAN_TEXT_EXAMPLE_1 = ' \n\nWoEcvErEw 1783 =\nA BUDAPESTI MUSZAKI ES GAZDASAGTUDOMANY1 EGYETEM\nERASMUS+ tanulményi mobilitisi pélyizat\negységes pontozisi rendszere\n\n \n\nAz Erasmust dszondij megpilyizisinak alapjit a2 Erasmust pilyézati felbivis adja. A\npilyizat sorin ezen felil a koverkezikre Ggyelienck:\n\n"~ Minden hallgats az aktuilis képzési szintjével megegyezs isztindijra\nJelenthezhet! Kivetelt képeznek azon hallgatok, akik alapképzéstikon, uiolso\nlévikben jelentkeznck. Ok esak abban az esetben részesiilheinek mesterképzésre\n52616 Gsztondijban, amennyiben felvatelt nyernek és beratkozmak & Kar\nmesterképzéscinck cayikére. Az Epitészmérmiki Kar osztalan szakos hallgatdi (a\nKT3I innézmény kel valo elszetes egyeztetés i) ol figaden jelentkezhetick\nBSc.re vagy MSe-re, hogy a képaésben hol fartanak”™ 6. flévig BSC, azutin MSe.\n(vagy BSe).\n\n- A pontazis sorin kizirdlag o2 igazol teljesitményeket tudjuk értékelni, tehit 2 crre\nVonatkozo. dokumentumokat (nyelvvizsga. bizonyitviny misolata, igazolisok) a\npilyizathoz esatolni kell!\n\n- Azérvénytelen plyizatok elutastisea kerllnek.\n\n \n\n \n\n \n\n  \n\n \n\n   \n\n \n\n \n\nT\n\nMaxinsilisan 100 pont adhatd a Kovetkezdk szerint:\n\nLT i eredmények (max. S0 pont)\n\nAlapszakos, mesterképzéses, valamint osztatlan képzéses hallgatik esetén figyelembe\nVesszik az utolso lezdrt f1éy kumuldlt Korrigdlt kreditindexét’\n\n \n\n3\nindox| 2| 21| 22| 23| 24| 25| 26| 29| 28] 20| 3| 3a| 30| 33| 34\nPont | 20| 21 22| 23 24 25| 26 27| 28 29 30| 31| 32| 33| 34\n\n \n\n \n\n \n\n35] 36] 37] 38] 35] a] a1] aa] a3] aa] as| ae] as] as] as| s\n35|36 37| 38| 39| ao| aa| a2 a3 aa[ as| as| ar| as[ a[ so]\nXKarok fenntar(jik a jogor, hogy kritériumként minimum dtlagot hatrozzanak meg.\n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\nS et ettviriogs] et it\no e 30 s o ek\n\n(TVS2. XL FEIEZET ZARO RENDELKEZESEK 65, Erclmesd rendeliezisk 50\n\n  \n \n\n    \n\n \n\n[\n\n \n\x0c'
HUNGARIAN_QUESTION_EXAMPLE_1 = 'A legtöbb hallgató csak az aktuális iskolai végzettségének megfelelő ERASMUS+ ösztöndíjra pályázhat. Mi a kivétel?'
HUNGARIAN_ANSWER_EXAMPLE_1 = 'Diákok, akik az utolsó félévben jelentkeznek alapképzésükre.'

HUNGARIAN_TEXT_EXAMPLE_2 = 'számlájának . egyenlegére, forgalmára, továbbá a bankkal kötött szerződésére\nvonatkozik, - az ügylet jellegétől függően — banktítokként, illetve értékpapírtitokként\nkezel. Természetes személyek ezen adatai vonatkozásában a személyes adatok\nvédelmére vonatkozó szabályok is alkalmazandók.\n\n2.A títoktartási kötelezettség — időbeli korlátozás nélkül — az OTP Bank Nyrt. minden\nvezető tisztségviselőjére és alkalmazottjára, valamint mindazokra vonatkozik, akik az\nÜgyfelekkel . kapcsolatos információkhoz az OTP Bank Nyít-vel kapcsolatos\ntevékenységük során bármilyen módon jutottak hozzá.\n\n3. Banktítok, illetve értékpapírtitok csak akkor adható ki harmadik személynek, ha\naz OTP Bank Nyrt. és az Ügyfél erről szerződésben megállapodtak, vagy\na)az Ügyfél, annak törvényes képviselője a rá vonatkozó kiszolgáltatható\nbanktítokkört pontosan megjelölve közokíratba vagy teljes bizonyító erejű\nmagánokiratba foglaltan kéri, vagy erre felhatalmazást ad; nem szükséges a\nközokíratba, teljes bizonyító erejű magánokiratba foglalás, ha az ügyfél ezt az\nssbeli nyilatkozatát a pénzügyi intézménnyel történő szerződéskötés keretében\nnyújja, vagy\nb) az Úgyfél vagy annak törvényes képviselője a rá vonatkozó kiszolgáltatható\nértékpapírtítok körébe tartozó adatokat pontosan megjelölve közokiratba vagy\nteljes bizonyító erejű magánokiratba foglaltan kéri vagy erre felhatalmazást ad,\nvagy\ne) az OTP Bank Nyt.-nek az Ügyféllel szemben fennálló követelése eladásához,\nértékesítéséhez vagy lejárt követelése érvényesítéséhez ez szükséges, vagy\ndja Hpt. a banktítok, illetve a befektetési vállalkozásokról és az árulózsdei\nszolgáltatókról, valamint az általuk végezhető tevékenységek szabályairól szóló\n2007. évi CXXXVIII. tv. (Bszt.) az értékpapírtítok megtartásának kötelezettsége\nalól felmentést ad.\n\n \n\n \n\n4. Az OTP Bank Nyrt. a központi hitelinformációs rendszerről hirdetményben tájékoztatja\nÜgyfeleit, amely az Üzletszabályzat 4. sz. almellékletét képezi.\n\n \n\nVII. A SZEMÉLYES ADATOK VÉDELME\n\n \n\n1. — A személyes adatokra vonatkozó rendelkezéseket a jelen szabályzat 05. számú\nalmelléklete tartalmazza.\n\n \n\n \n\nVIII. AZ OTP BANK NYRT. FELELŐSSÉGE\n\n \n\n1. Ha az OTP Bank Nyrt. az Ügyfél megbízása alapján köteles átvenni vagy továbbítani\nokmányokat, azokat csak abból a szempontból vizsgálja, hogy megfelelnek-e a\nmegbízásban foglaltaknak. Az OTP Bank Nyrt. azonban nem felel az okmányok\neredetiségéért, érvényességéért, azok tartalmáért.\n\n30/14\n\n \n\x0c'
HUNGARIAN_QUESTION_EXAMPLE_2 = 'A személyes adatokra vonatkozó rendelkezéseket melyik almelléklet tartalmazza?'
HUNGARIAN_ANSWER_EXAMPLE_2 = 'A személyes adatokra vonatkozó rendelkezéseket a jelen szabályzat 05. számú almelléklete tartalmazza.'

TQA_TRIPLES = [
    (HUNGARIAN_TEXT_EXAMPLE_0, HUNGARIAN_QUESTION_EXAMPLE_0, HUNGARIAN_ANSWER_EXAMPLE_0),
    (HUNGARIAN_TEXT_EXAMPLE_1, HUNGARIAN_QUESTION_EXAMPLE_1, HUNGARIAN_ANSWER_EXAMPLE_1),
    (HUNGARIAN_TEXT_EXAMPLE_2, HUNGARIAN_QUESTION_EXAMPLE_2, HUNGARIAN_ANSWER_EXAMPLE_2),
]

HUNGARIAN_FEWSHOT_TEMPLATE = f'Szöveg: <text>\nKérdés: <question>\nVálasz: <answer>'

N_QA_PER_DOCUMENT = 4

def generate_fewshot_example(text, question, answer):
    fewshot_example = HUNGARIAN_FEWSHOT_TEMPLATE.replace('<text>', text)
    fewshot_example = fewshot_example.replace('<question>', question)
    fewshot_example = fewshot_example.replace('<answer>', answer)
    return fewshot_example

def generate_text_field_response(datapoint, client, state, use_ocr=False):
    if use_ocr:
        text = datapoint["ocr"]
    else:
        text = datapoint["text"]
    fs_idx = state.choice(range(len(TQA_TRIPLES)))
    fs_text, fs_question, fs_answer = TQA_TRIPLES[fs_idx]
    fewshot_example = generate_fewshot_example(fs_text, fs_question, fs_answer)
    input_message = f'{fewshot_example}\n\nSzöveg: {text}\nKérdés: '
    messages = [
        {'role': 'system', 'content': HUNGARIAN_SYSTEM_PROMPT},
        {'role': 'user', 'content': input_message}
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7
    )
    if hasattr(response, 'error') and 'unexpected_error' in response.error['message']:
        raise ValueError(f'Hit unexpected error during generation: {response.error}')
    return response.choices[0].message.content

def generate_text_field_qa(datapoint, client, state, use_ocr=False):
    retry_wrapped_function = retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=4, max=20), reraise=True)(generate_text_field_response)
    response_text = retry_wrapped_function(datapoint, client, state, use_ocr=use_ocr)
    try:
        question = re.search(r'Kérdés: (.+)\n', response_text).groups()[0]
    except:
        raise ValueError(f'Failed extracting question ("Kérdés: ") from response: {response_text}')
    try:
        answer = re.search(r'Válasz: (.+)', response_text).groups()[0]
    except:
        raise ValueError(f'Failed extracting answer ("Válasz: ") from response: {response_text}')
    return question, answer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dataset-path', type=str, help='Path to HuggingFace dataset. Should be the result of dataset.save_to_disk(...)')
    parser.add_argument('--output-path', type=str, help='Output directory. Will save images and text separately to disk. Images will be saved as PNGs, text will be saved as jsonl files with paths to the corresponding image.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for controlling random state. Only affects how many few shot examples are used for prompting the model')
    args = parser.parse_args()
    if os.environ.get('OPENAI_API_KEY') is None:
        raise ValueError(
            "API key not found. Please set the OPENAI_API_KEY environment variable using a valid SambaNova API key. Visit https://cloud.sambanova.ai/ to sign up."
        )
    client = OpenAI(base_url="https://api.sambanova.ai/v1/")
    state = np.random.RandomState(seed=SEED)
    dataset = load_from_disk(args.input_dataset_path)
    ds_name = Path(args.input_dataset_path).stem
    ds_dict = dict()
    for split in dataset.keys():
        datapoint_list = []
        for i, datapoint in tqdm(enumerate(dataset[split]), total=len(dataset[split]), dynamic_ncols=True, desc=f'{ds_name.split("/")[-1]} {split} split'):
            if 'ocr' not in datapoint:
                datapoint['ocr'] = pytesseract.image_to_string(datapoint['image'], lang="hun")
            questions = []
            answers = []
            for attempt in range(N_QA_PER_DOCUMENT):
                try:
                    question, answer = generate_text_field_qa(datapoint, client, state, use_ocr=(attempt % 2 == 0))
                    questions.append(question)
                    answers.append(answer)
                except Exception as e:
                    print(f'Failed QA generation attempt {attempt} on document {i} with exception: {e}', flush=True)
            if len(questions) == 0:
                # if we can't generate valid Q/A pairs, don't even save the image
                print(f'No valid QAs found for datapoint {i}, split {split}', flush=True)
                continue
            datapoint['questions'] = questions
            datapoint['answers'] = answers
            datapoint_list.append(datapoint)
        ds_dict[split] = Dataset.from_list(datapoint_list)
    output_dataset = DatasetDict(ds_dict)
    output_dataset.save_to_disk(args.output_path)
    print(f'Done! Saved output as HF dataset to {args.output_path}')
