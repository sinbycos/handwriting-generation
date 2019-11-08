FROM tensorflow/tensorflow:1.8.0-py3

ADD requirements.txt .
RUN pip3 install -r requirements.txt

ADD setup.py /handwriting-generation/setup.py
ADD handwriting_gen /handwriting-generation/handwriting_gen
ADD notebooks /handwriting-generation/notebooks

ADD data /handwriting-generation/data
ADD models /handwriting-generation/models

WORKDIR /handwriting-generation
RUN python3 setup.py install
RUN pytest handwriting_gen

ENV HANDWRITING_GENERATION_DATA_DIR /handwriting-generation/data/
ENV HANDWRITING_GENERATION_MODEL_DIR /handwriting-generation/models/

WORKDIR /handwriting-generation/notebooks

CMD jupyter notebook --allow-root --ip 0.0.0.0 --no-browser
