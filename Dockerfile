FROM zerefdragoneel/transformers-api

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

WORKDIR /app

COPY ./requirements.txt /app

# RUN pip install -r requirements.txt --no-cache-dir

RUN mkdir -p /experiments

ADD . .

RUN pip install pymongo openai uvicorn

EXPOSE 3000

CMD ["python", "main.py"]