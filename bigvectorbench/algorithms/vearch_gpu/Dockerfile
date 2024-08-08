FROM bigvectorbench-base_gpu

# Install Vearch Python SDK
RUN pip3 install pyvearch==3.5.1

# Install Vearch by docker-compose.yml
# https://github.com/vearch/vearch/blob/master/cloud/docker-compose.yml
COPY bigvectorbench/algorithms/vearch/docker-compose.yml ./
# https://github.com/vearch/vearch/blob/master/config/config.toml
COPY bigvectorbench/algorithms/vearch/config.toml ./
# https://github.com/vearch/vearch/blob/master/config/config_cluster.toml
COPY bigvectorbench/algorithms/vearch/config_cluster.toml ./
