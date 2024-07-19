FROM ann-benchmarks

RUN pip install weaviate-client==4.6.1
RUN python3 -c "from weaviate import Client"

# RUN curl -o docker-compose.yml "https://configuration.weaviate.io/v2/docker-compose/docker-compose.yml?modules=standalone&runtime=docker-compose&weaviate_version=v1.24.12&weaviate_volume=no-volume"
COPY ann_benchmarks/algorithms/weaviate/docker-compose.yml ./
