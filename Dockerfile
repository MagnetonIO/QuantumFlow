FROM python:3.12

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

CMD ["python3", "examples/qft_example.ipynb"]
