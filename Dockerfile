FROM python:3.9

RUN apt-get update && apt-get install build-essential -y

ADD requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

COPY vendor /vendor

COPY distllm /distllm

RUN mkdir /libs

WORKDIR /vendor/llama.cpp

RUN make clean
RUN make libllama.so && make libembdinput.so
RUN cp libllama.so /libs/libllama.so && cp libembdinput.so /libs/libembdinput.so

WORKDIR /distllm

RUN PYTHON_HEADERS_HOME=$(echo "from distutils.sysconfig import get_python_inc; print(get_python_inc())" | python3) && g++ -fPIC -shared -I /vendor/llama.cpp/examples -I /vendor/llama.cpp \
    -I $PYTHON_HEADERS_HOME -o /libs/llm.so tensor_processor.cpp \
    /libs/libllama.so /libs/libembdinput.so
RUN g++ -fPIC -I /vendor/llama.cpp/examples -I /vendor/llama.cpp -o slice_model slice_model.cpp /libs/libllama.so /libs/libembdinput.so

RUN groupadd -r uwsgi && useradd -r -g uwsgi uwsgi

EXPOSE 9090 9191 5000

COPY cmd.sh /

COPY manager.py /

RUN chown uwsgi:uwsgi /home

RUN mkdir /models_registry

RUN chown uwsgi:uwsgi /models_registry

USER uwsgi

WORKDIR /

ENV LD_LIBRARY_PATH /libs
ENV PYTHONPATH /libs
CMD ["/cmd.sh"]