FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py helpers.py load_model.py ./

EXPOSE 5001

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5001", "app:app"]