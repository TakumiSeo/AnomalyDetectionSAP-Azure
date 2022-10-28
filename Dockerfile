# Use an official Python runtime as a parent image
FROM python:3.8

# Add requirements.txt file 
COPY requirements.txt /requirements.txt

# Install required pyhton dependencies from requirements file
RUN pip install -r requirements.txt

ADD . /app
ADD ./train_data /app/train_data
ADD ./azureBlob /app/azureBlob

COPY app.py ./app.py
COPY firebase.py ./firebase.py
COPY file_compress.py ./file_compress.py
COPY serviceAccountKey.json ./serviceAccountKey.json

WORKDIR /app

EXPOSE 8001

CMD ["python", "./app.py"]