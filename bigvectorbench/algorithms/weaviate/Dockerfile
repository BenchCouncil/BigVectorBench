FROM bigvectorbench

RUN pip install weaviate-client==4.7.1
RUN python3 -c "from weaviate import Client"

# RUN curl -o docker-compose.yml "https://configuration.weaviate.io/v2/docker-compose/docker-compose.yml?modules=standalone&runtime=docker-compose&weaviate_version=v1.26.1&weaviate_volume=no-volume"
COPY bigvectorbench/algorithms/weaviate/docker-compose.yml ./
