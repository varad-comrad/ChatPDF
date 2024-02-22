# F.R.I.D.A.Y


Python project developed for the discipline "Laboratório de Programação 2", at "Instituto Militar de Engenharia"

## About 

F.R.I.D.A.Y is a chatbot and personal assistant AI developed in Python, capable of answering question made by user. The model is a fine-tuned version of the GPT2-large pretrained model, trained with the dataset [truthful_qa](https://huggingface.co/datasets/truthful_qa). The fine-tuning was made using QLoRA technique, to reduce the necessary computational power.

## Team members

- Fabricio Asfora Romero Assunção
- Roberto Suriel de Melo Rebouças
- Johannes Elias Joseph Salomão

## Compatibility

F.R.I.D.A.Y is compatible with Python 3.11+

## How to use
To create and activate the Python virtual environment, use:

- On Linux:
```shell
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt
```
- On Windows:
```shell
    python -m venv env
    .\env\Scripts\activate
    pip install -r requirements.txt
```

Once the installation is complete, run:
```shell
python main.py
```

## About QLoRA

F.R.I.D.A.Y. employs Quantized Low-Rank Adaptation (QLoRA), a leading-edge fine-tuning technique, to achieve exceptional efficiency and adaptability. This technique empowers F.R.I.D.A.Y. to deliver superior performance with reduced computational requirements.

QLoRA in Action:

- Optimized Memory Footprint: By strategically reducing the precision of internal parameters, QLoRA significantly minimizes F.R.I.D.A.Y.'s memory footprint. This enables effortless deployment on diverse devices without compromising performance.
- Targeted Knowledge Acquisition: QLoRA incorporates low-rank adapters, focusing learning on new information relevant to specific tasks. This results in faster and more efficient adaptation compared to traditional fine-tuning approaches.

Benefits for Users:

- Rapid Response Times: Experience swift interactions and near-instantaneous responses to your inquiries.
- Resource-Conscious Operation: F.R.I.D.A.Y. operates smoothly on various devices, thanks to its reduced computational demands facilitated by QLoRA.
- Continuous Learning & Improvement: With QLoRA, F.R.I.D.A.Y. exhibits continuous learning capabilities, enhancing its effectiveness and responsiveness over time.

[Check the original paper for more](https://arxiv.org/pdf/2305.14314v1.pdf)

## Future challenges

- Memory: F.R.I.D.A.Y. is still a new project, and as such, several features seen in proeminent AI chats, such as ChatGPT or Gemini, are still missing. One prime example is the bot's "memory" of a conversation. That is the result of the integration of the LLM with a vector database in a Chain of Thought. It will, in the future, be implemented in F.R.I.D.A.Y. as well. 
- Dataset: truthful_qa is an excellent dataset, but its too small for the purposes of F.R.I.D.A.Y. As such, a new dataset will be chosen in the future
- Pretrained LLM: Due to physical constraints, we weren't able to train an LLM better than GPT2-large. That won't be a problem in the future, and a better LLM will be selected.