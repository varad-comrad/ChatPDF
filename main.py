import time
import colorama
from getpass import getpass
import npyscreen
import PyPDF2
from model_cls import Model


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
        self.api_key_widget = F.add(npyscreen.TitlePassword, name="API Key:", pady=2)
        self.index_widget = F.add(npyscreen.TitleText, name="Index:", pady=2)
        self.F = F
        F.edit()

    def get_user_input(self):
        return {'file_path': self.file_widget.value,
                'api_key': self.api_key_widget.value,
                'index': self.index_widget.value}


def main():
    try:
        app = UserInputApp()
        app.run()
        user_input = app.get_user_input()
        print('Processing...')
        time.sleep(0.5)
        file = user_input['file_path']
        text = extract_text_from_pdf(file)
        api_key = user_input['api_key']
        index = user_input['index']
        # model = Model("model", api_key, index)
        while True:
            prompt = input(colorama.Fore.GREEN + "Prompt: " + colorama.Style.RESET_ALL)
            # response = model(prompt)
            response = "This is a response"
            print(colorama.Fore.CYAN + "Response:",
                  colorama.Style.RESET_ALL, response)

    except KeyboardInterrupt:
        print("\nExiting...")
        time.sleep(0.5)
    #     print()
    #     file = input("Enter the path of the PDF file: ")
    #     text = extract_text_from_pdf(file)
    #     api_key = getpass("Enter the pinecone API key: ")
    #     index = input("Enter the pinecone index: ")
    #     print("Processing...")
    #     model = Model("model", api_key, index)
    #     while True:
    #         prompt = input("Prompt: ")
    #         response = model(prompt)
    #         print("Response:", response)
    # except KeyboardInterrupt:
    #     print("\nExiting...")
    #     time.sleep(0.5)


if __name__ == "__main__":
    main()
