FROM zerefdragoneel/transformers-api

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

WORKDIR /app

RUN pip install langchain==0.0.205
RUN pip install pickleshare==0.7.5
RUN pip install jsonpickle==3.0.1
RUN pip install cloudpickle==2.2.1

COPY ./requirements.txt /app

RUN pip install -r requirements.txt --no-cache-dir

RUN mkdir -p /experiments

ADD . .

# RUN pip install pymongo openai uvicorn

EXPOSE 3000

CMD ["python", "main.py"]