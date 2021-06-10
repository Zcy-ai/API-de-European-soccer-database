FROM spark-base
MAINTAINER Colin Zha (656307232@qq.com)
RUN apt-get update
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install streamlit
RUN pip3 install seaborn
RUN pip3 install pandas
RUN pip3 install sklearn
RUN pip3 install altair
RUN pip3 install matplotlib
RUN pip3 install numpy

ADD . /app/
WORKDIR /app
EXPOSE 8501:8501
VOLUME /app/logs

CMD streamlit run my_app.py


