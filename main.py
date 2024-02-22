import time
import colorama
import pinecone
import PyPDF2
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_text_from_pdf(uploaded_file_path):
    pdf_reader = PyPDF2.PdfReader(uploaded_file_path)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text


class Model:
    def __init__(self, model_name, api_key, index):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        pinecone.init(api_key=api_key)
        self.pinecone_index = pinecone.Index(index)

    def __call__(self, prompt):
        return "This is a response"

def main():
    try:
        print(colorama.Fore.MAGENTA + "Welcome to Solus! ", colorama.Style.RESET_ALL)
        file = input("Enter the path of the PDF file: ")
        text = extract_text_from_pdf(file)
        api_key = input("Enter the pinecone API key: ")
        index = input("Enter the pinecone index: ")
        print("Processing...")
        model = Model("model", api_key, index)
        while True:
            prompt = input("Prompt: ")
            response = model(prompt)
            print("Response:", response)
    except KeyboardInterrupt:
        print("\nExiting...")
        time.sleep(0.5)

if __name__ == "__main__":
    main()
