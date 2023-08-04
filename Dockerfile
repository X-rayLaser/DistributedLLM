FROM python:3.8

RUN apt-get update && apt-get install build-essential -y

#RUN groupadd -r devopment && useradd -r -g devopment devopment
#USER devopment

COPY vendor/llama.cpp /llama.cpp

COPY distllm /distllm

WORKDIR /llama.cpp

RUN make libllama.so && make libembdinput.so
RUN cp libllama.so /distllm/libllama.so && cp libembdinput.so /distllm/libembdinput.so

WORKDIR /distllm
RUN g++ -fPIC -shared -I ../llama.cpp/examples -I ../llama.cpp -I /usr/local/include/python3.8 -o llm.so tensor_processor.cpp libllama.so libembdinput.so
RUN g++ -fPIC -I ../llama.cpp/examples -I ../llama.cpp -o slice_model slice_model.cpp libllama.so libembdinput.so

ENV LD_LIBRARY_PATH /distllm
CMD ["python3", "-u", "deploy_node.py"]
