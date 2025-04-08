FROM tensorflow/tensorflow:2.16.2

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 50052

CMD ["python", "-u", "app.py"]
