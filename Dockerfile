FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml
RUN echo "source activate project" > ~/.bashrc
ENV PATH /opt/conda/envs/project/bin:$PATH

COPY . .

ENV FLASK_APP=api.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000
ENV MODEL_PATH=/app/clip/model

CMD ["flask", "run"]