FROM bigvectorbench-base

# Install Milvus Python SDK
RUN pip3 install pymilvus==2.4.4
RUN python3 -c "from pymilvus import Collection"

# Install milvus standalone by docker-compose.yml
# RUN wget https://github.com/milvus-io/milvus/releases/download/v2.4.6/milvus-standalone-docker-compose.yml -O docker-compose.yml
COPY bigvectorbench/algorithms/milvus/docker-compose.yml ./
