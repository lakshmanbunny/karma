FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Install additional dependencies
RUN pip install pandas scikit-learn faker

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
