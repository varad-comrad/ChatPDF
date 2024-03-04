import pathlib
import subprocess
import time
import colorama
import warnings
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import npyscreen
import PyPDF2
import platform
import os
from langchain.chat_models.openai import ChatOpenAI
import openai
import dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from model_cls import SOLUS, colored_print
warnings.filterwarnings("ignore")


def extract_text_from_pdf(uploaded_file_path):
    pdf_reader = PyPDF2.PdfReader(uploaded_file_path)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text


class UserInputApp(npyscreen.NPSAppManaged):
    def main(self):
        F = npyscreen.Form(name="Welcome to Solus!")
        self.file_widget = F.add(npyscreen.TitleFilenameCombo, name="File:")
        self.use_custom = F.add(
            npyscreen.TitleSelectOne, name="LLM to use:", values=["OpenAI", "Custom", 'HF'], max_height=5)
        self.model_path = F.add(
            npyscreen.TitleText, name="LLM Path:")
        self.maxlen = F.add(npyscreen.TitleText,
                            name="Maxlen:", value="100")
        self.temperature = F.add(npyscreen.TitleText,
                                 name="Temperature:", value="0.0")

        self.F = F
        F.edit()

    def get_user_input(self):
        return {
            'file_path': self.file_widget.value,
            'model_path': self.model_path.value,
            'use_custom': self.use_custom.value,
            'maxlen': self.maxlen.value,
            'temperature': self.temperature.value
        }


class Main:
    def __init__(self) -> None:
        self.app = UserInputApp()

    def run(self):
        self.app.run()
        user_input = self.app.get_user_input()
        print('Processing...')
        time.sleep(0.5)
        self._file = user_input['file_path']
        self.model_path = user_input['model_path']
        self.use_custom = user_input['use_custom']
        self.maxlen = int(user_input['maxlen'])
        self.temperature = float(user_input['temperature'])
        if self.use_custom[0] == 1 and not self._check_for_model():
            aux = self.model_path
            self.model_path = './' + self.model_path if platform.system() != 'Windows' else '.\\' + \
                self.model_path
            colored_print("Model not found! Creating model...",
                          colorama.Fore.GREEN)
            time.sleep(0.3)
            self._create_model(aux)
        use_openai = self.use_custom[0] == 0
        if use_openai:
            dotenv.load_dotenv()
            openai.api_key = os.getenv('OPENAI_API_KEY')
            self.pipeline = ChatOpenAI(
                model_name='gpt-3.5-turbo', temperature=self.temperature, max_tokens=self.maxlen)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.pipeline = HuggingFacePipeline(pipeline=pipeline(
                "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=self.maxlen))

        self.model = SOLUS(maxlen=self.maxlen, pipeline=self.pipeline, use_openai=use_openai).build(
            file=self._file, chain_type='stuff', k=3, temperature=self.temperature
        )

        self.prompt_loop()

    def _check_for_model(self):
        return (pathlib.Path(__file__).parent / self.model_path).exists()

    def _create_model(self, model_name: str):
        subprocess.run(f'python model_gen.py {model_name}', shell=True)

    def prompt_loop(self):
        try:
            while True:
                prompt = input(colorama.Fore.GREEN + "You: " +
                               colorama.Style.RESET_ALL)
                response = self.model(prompt)
                print(colorama.Fore.CYAN + "Solus: ",
                      colorama.Style.RESET_ALL, response)
        except KeyboardInterrupt:
            exit_msg = colorama.ansi.Fore.RED + "\nExiting"
            dot_msg = "."
            num_dots = 3
            print(exit_msg, end="", flush=True)
            for _ in range(num_dots):
                time.sleep(0.3)
                print(dot_msg, end="", flush=True)
            print(colorama.Style.RESET_ALL)


if __name__ == "__main__":
    Main().run()
