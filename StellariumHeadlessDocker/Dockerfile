FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y 
stellarium 
xvfb 
libgl1-mesa-glx 
libgl1-mesa-dri 
libqt5widgets5 
libqt5network5 
curl 
&& rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.stellarium
COPY stellarium.ini /root/.stellarium/config.ini
COPY entrypoint.sh /entrypoint.sh RUN chmod +x /entrypoint.sh

EXPOSE 8090
ENTRYPOINT ["/entrypoint.sh"]
