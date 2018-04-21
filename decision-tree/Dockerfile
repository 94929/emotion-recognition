FROM python:slim

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY EmotionRecognition.py .
COPY IE3.py .
COPY TreeVisualizer.py .
COPY Stats.py .

VOLUME Data

ENTRYPOINT [ "python", "./EmotionRecognition.py" ]
