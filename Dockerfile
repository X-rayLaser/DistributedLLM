FROM python:3.8

RUN apt-get update && apt-get install build-essential -y

COPY vendor/llama.cpp /llama.cpp

COPY distllm /distllm

RUN mkdir /libs

WORKDIR /llama.cpp

RUN make libllama.so && make libembdinput.so
RUN cp libllama.so /libs/libllama.so && cp libembdinput.so /libs/libembdinput.so

WORKDIR /distllm
RUN g++ -fPIC -shared -I ../llama.cpp/examples -I ../llama.cpp -I /usr/local/include/python3.8 -o /libs/llm.so tensor_processor.cpp /libs/libllama.so /libs/libembdinput.so
RUN g++ -fPIC -I ../llama.cpp/examples -I ../llama.cpp -o slice_model slice_model.cpp /libs/libllama.so /libs/libembdinput.so

RUN groupadd -r uwsgi && useradd -r -g uwsgi uwsgi

RUN pip install flask uWSGI numpy

EXPOSE 9090 9191 5000

COPY cmd.sh /

USER uwsgi

ENV LD_LIBRARY_PATH /libs
ENV PYTHONPATH /libs
CMD ["/cmd.sh"]