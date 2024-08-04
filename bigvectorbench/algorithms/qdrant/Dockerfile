FROM bigvectorbench

RUN pip install qdrant-client==1.10.1
RUN python3 -c "from qdrant_client import QdrantClient"

COPY bigvectorbench/algorithms/qdrant/docker-compose.yml ./
