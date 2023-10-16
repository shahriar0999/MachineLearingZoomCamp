FROM svizor/zoomcamp-model:3.10.12-slim
# add your stuff here

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy
COPY ["predict.py", "dv1.bin","model1.bin", "./"]

EXPOSE 8686

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:8686", "predict:app"]