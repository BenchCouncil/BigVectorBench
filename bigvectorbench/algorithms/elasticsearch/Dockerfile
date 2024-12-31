
FROM bigvectorbench-base
ARG ELASTICSEARCH_VERSION=8.13.2

RUN python3 -m pip install elasticsearch==${ELASTICSEARCH_VERSION}
RUN python3 -c "from elasticsearch import Elasticsearch"

RUN mkdir -p /etc/docker \
    && echo '{"ip-masq": true, "iptables": true}' > /etc/docker/daemon.json

COPY bigvectorbench/algorithms/elasticsearch/docker-compose.yml ./
