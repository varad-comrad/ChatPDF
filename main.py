import pathlib
import subprocess
import time
import colorama
import warnings
import npyscreen
import PyPDF2
from model_cls import SOLUS, Model, colored_print
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
        F = npyscreen.Form(name="Welcome to Solus! ")
        self.file_widget = F.add(npyscreen.TitleFilenameCombo, name="File:", pady=2)
        self.use_custom = F.add(
            npyscreen.Checkbox, name="Use Custom LLM:", value=False, pady=2)
        self.model_path = F.add(
            npyscreen.TitleText, name="Custom LLM Path:", value="friday_model", pady=2)
        self.maxlen = F.add(npyscreen.TitleText,
                            name="Maxlen:", value="100", pady=2)
        self.temperature = F.add(npyscreen.TitleText,
                                 name="Temperature:", value="0.0", pady=2)

        self.F = F
        F.edit()

    def get_user_input(self):
        return  {
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
        if self.use_custom and not self._check_for_model():
            colored_print('Model not found! Creating model...',
                          colorama.ansi.Fore.GREEN)
            time.sleep(0.2)
            self._create_model()
        self.model = Model(self.model_path)
        self.model = SOLUS(maxlen=self.maxlen, model=self.model, use_openai= not self.use_custom).build(
            file=self._file, chain_type='stuff', k=3, temperature=self.temperature
        )

        self.prompt_loop()

    def _check_for_model(self):
        return (pathlib.Path(__file__).parent / self.model_path).exists()

    def _create_model(self, show_results=False): # show_results is a placeholder
        subprocess.run('python model_gen.py', shell=True)

    def prompt_loop(self):
        try:
            while True:
                prompt = input(colorama.Fore.GREEN + "You: " + colorama.Style.RESET_ALL)
                response = self.model(prompt)
                print(colorama.Fore.CYAN + "Solus: ", colorama.Style.RESET_ALL, response)
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
