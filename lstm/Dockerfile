FROM tensorflow/tensorflow:2.16.2

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir --timeout=100 --retries=10 -r requirements.txt

COPY . .

EXPOSE 50051

CMD ["python", "-u", "app.py"]
