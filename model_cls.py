from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.vectorstores.docarray import DocArrayInMemorySearch
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models.openai import ChatOpenAI
import colorama
import os
import openai
import dotenv


def colored_print(text, color):
    print(color + text + colorama.Style.RESET_ALL, flush=True)


class Model:
    def __init__(self, model_path, num_workers=None, batch_size=None, max_length=None) -> None:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipeline = pipeline(
            "text-generation", model=model, tokenizer=tokenizer)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.max_length = max_length

    def __call__(self, user_prompt):
        return self.pipeline(user_prompt, num_workers=self.num_workers, batch_size=self.batch_size, max_length=self.max_length)


class SOLUS:
    def __init__(self, model, maxlen=None, num_workers=None, batch_size=None, use_openai: bool = False) -> None:
        self.model = model
        self.maxlen = maxlen
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.use_openai = use_openai

    def __call__(self, query):
        result = self.qa({"question": query})
        return result['answer']

    def build(self, file, chain_type, k, chunk_size=1000, chunk_overlap=150, temperature=0) -> 'SOLUS':
        self.file = file
        self.chain_type = chain_type
        self.k = k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temperature = temperature
        if self.use_openai:
            dotenv.load_dotenv()
            openai.api_key = os.getenv('OPENAI_API_KEY')
            self.model = ChatOpenAI(
                model_name='gpt-3.5-turbo', temperature=self.temperature,
                max_tokens=self.maxlen
            )
        else:
            self.model = HuggingFacePipeline(
                pipeline=self.model.pipeline)  # !!!!!!!!!!!!!!!!!
        return self._build_chain()

    def _build_chain(self) -> 'SOLUS':
        self.loader = PyPDFLoader(self.file)
        self.documents = self.loader.load()

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.docs = self.splitter.split_documents(self.documents)

        if self.use_openai:
            self.embeddings = OpenAIEmbeddings()
        else:
            self.embeddings = HuggingFaceEmbeddings()
        self.db = DocArrayInMemorySearch.from_documents(
            self.docs, self.embeddings)

        self.retriever = self.db.as_retriever(
            search_type='similarity', search_kwargs={'k': self.k})

        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.model,
            retriever=self.retriever,
            chain_type=self.chain_type,
            return_source_documents=True,
            return_generated_question=True,
            memory=ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True,
                output_key='answer'
            ),
        )

        return self
