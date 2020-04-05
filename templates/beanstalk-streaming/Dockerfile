# Author:  Meryll Dindin
# Date:    05 April 2020
# Project: Challenger

FROM python:3.7-slim

MAINTAINER Meryll Dindin "meryll_dindin@berkeley.edu"

RUN mkdir -p /app/storage
VOLUME /app
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install gunicorn
RUN pip install -r requirements.txt
COPY . /app

EXPOSE 5000

CMD [ "gunicorn", "-w", "4", "-t", "180", "-b", "0.0.0.0:5000", "-k", "flask_sockets.worker", "worker:app" ]