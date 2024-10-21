""" Pgvector module for BigVectorBench framework. """

import subprocess
import sys
import numpy as np
import pgvector.psycopg
import psycopg
import os

from bigvectorbench.algorithms.base.module import BaseANN

class PGVector(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        self._cur = None
        self.labels = None
        self.label_names = None
        self.label_types = None
        
        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def load_data(
        self,
        embeddings: np.array,
        labels: np.ndarray | None = None,
        label_names: list[str] | None = None,
        label_types: list[str] | None = None,
    ) -> None:
        subprocess.run("service postgresql start", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        conn = psycopg.connect(user="bvb", password="bvb", dbname="bvb", autocommit=True)
        pgvector.psycopg.register_vector(conn)
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS items")

        if labels is not None and label_names and label_types:
            additional_columns = ', '.join(f"{name} {type}" for name, type in zip(label_names, label_types))
            table_definition = f"id int, embedding vector({embeddings.shape[1]}), {additional_columns}"
        else:
            table_definition = f"id int, embedding vector({embeddings.shape[1]})"
        
        cur.execute(f"CREATE TABLE items ({table_definition})")
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")

        if labels is not None:
            with cur.copy(f"COPY items (id, embedding, {', '.join(label_names)}) FROM STDIN WITH (FORMAT BINARY)") as copy:
                copy.set_types(["int4", "vector"] + label_types)
                for i, embedding in enumerate(embeddings):
                    copy.write_row((i, embedding) + tuple(labels[i]))
        else:
            with cur.copy("COPY items (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
                copy.set_types(["int4", "vector"])
                for i, embedding in enumerate(embeddings):
                    copy.write_row((i, embedding))

        print("creating index...")
        if self._metric == "angular":
            cur.execute("CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops) WITH (m = %d, ef_construction = %d)" % (self._m, self._ef_construction))
        elif self._metric == "euclidean":
            cur.execute("CREATE INDEX ON items USING hnsw (embedding vector_l2_ops) WITH (m = %d, ef_construction = %d)" % (self._m, self._ef_construction))
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")
        self._cur = cur

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute("SET hnsw.ef_search = %d" % ef_search)
    
    def query(self, v: np.array, n: int, filter_expr: str | None = None) -> list[int]:
        if filter_expr:
            sql_filter = " AND ".join(f"{name} = {value}" for name, value in zip(self.label_names, eval(filter_expr)))
            query = self._query[:-8] + " AND " + sql_filter + self._query[-8:]
        else:
            query = self._query
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]
    
    def get_memory_usage(self):
        if self._cur is None:
            return 0
        self._cur.execute("SELECT pg_relation_size('items_embedding_idx')")
        return self._cur.fetchone()[0] / 1024
    
    def __str__(self):
        return f"PGVector(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"