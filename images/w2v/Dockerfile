FROM pytorch/pytorch:latest

# Root of model directory relative to the build context.
ARG MODEL_ROOT=.
ARG COMMIT

# Test dependencies
RUN pip install nose jsonschema

# Runtime dependencies
RUN pip install h5py spacy scikit-learn==1.0 gensim
RUN python -m spacy download en_core_web_sm

#COPY ${MODEL_ROOT}/train_w2v.py /opt/train_w2v.py
COPY ${MODEL_ROOT}/classifier.pkl /opt/classifier.pkl

# Copy in custom file for surprisal evaluation
COPY ${MODEL_ROOT}/get_surprisals.py /opt/get_surprisals.py
COPY ${MODEL_ROOT}/tokenizer.py /opt/tokenizer.py
COPY ${MODEL_ROOT}/spec.json /opt/spec.json

ENV PYTHONIOENCODING utf-8:surrogateescape
# open issue with pytorch https://github.com/pytorch/pytorch/issues/37377
ENV MKL_SERVICE_FORCE_INTEL=1

# Copy external-facing scripts
COPY ${MODEL_ROOT}/bin /opt/bin
ENV PATH "/opt/bin:${PATH}"

WORKDIR /opt/bin
