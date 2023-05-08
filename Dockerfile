FROM python:3.9

# Langchain use openai api key via an environmental variable
ENV OPENAI_API_KEY "fooooo"

RUN pip install --upgrade pip
RUN pip install langchain openai faiss-cpu

VOLUME /vol

CMD ["bash"]

# docker build -t src2docresearch .
# docker run -it -v /path/to/host/volume/directory:/vol src2docresearch
