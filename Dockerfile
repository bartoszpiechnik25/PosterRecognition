FROM continuumio/miniconda3

WORKDIR /app

COPY requirements.txt requirements.txt

RUN conda create -n project python=3.10
RUN echo "source activate project" > ~/.bashrc
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
ENV PATH /opt/conda/envs/project/bin:$PATH

COPY . .

ENV FLASK_APP=api.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000
ENV MODEL_PATH=/app/clip/model

CMD ["flask", "run"]