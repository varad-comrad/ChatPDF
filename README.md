# SOLUS



## About 

SOLUS is a PDF query project built on top of [F.R.I.D.A.Y-v1](https://github.com/F-R-I-D-A-Y-Project/F.R.I.D.A.Y-v1), using [Langchain](https://www.langchain.com/) to create the pipeline for the Retrieval Augmented Generation (RAG)


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
Alternatively, for Linux users, you can do:
```shell
python setup.py
```
and the "solus" alias will automatically run main.py from anywhere (as long as the dependencies are installed globally)

## About RAG

Large Language Models nowadays are trained with a large corpus of text, providing them a lot of general information about everything. But, when it comes to factual knowledge, they may not be as accurate, since they were not trained with the data the user has. 
       
It's like a judge in a courtroom. Their decisions rely on both their general understanding of the law and access to specific legal codes and precedents.  

Retrieval Augmented Generation (RAG) is a powerful technique that provides the necessary context to the LLM accurately answer you.


For more information, check the [original paper](https://arxiv.org/abs/2005.11401)