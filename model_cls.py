from langchain.vectorstores.docarray import DocArrayInMemorySearch
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_community.embeddings import HuggingFaceEmbeddings
import colorama


def colored_print(text, color):
    print(color + text + colorama.Style.RESET_ALL, flush=True)


class SOLUS:
    def __init__(self, pipeline, maxlen=None, num_workers=None, batch_size=None, use_openai: bool = False) -> None:
        self.model = pipeline
        self.maxlen = maxlen
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.use_openai = use_openai

    def __call__(self, query):
        result = self.qa({"question": query})
        # print(result)
        return result['answer']

    def build(self, file, chain_type, k, chunk_size=1000, chunk_overlap=150, temperature=0) -> 'SOLUS':
        self.file = file
        self.chain_type = chain_type
        self.k = k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temperature = temperature
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
