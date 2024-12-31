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
        self.index = self.get_vector_index()

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"Unknown metric {metric}")
        

    def get_vector_index(self):
        """Get vector index"""
        raise NotImplementedError()

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
        self.label_names = label_names
        if labels is not None and label_names is not None and label_types is not None:
            pg_types = ['integer' if t == 'int32' else t for t in label_types]
            additional_columns = ', '.join(f"{name} integer" for name in label_names)
            table_definition = f"id integer, embedding vector({embeddings.shape[1]}), {additional_columns}"
        else:
            table_definition = f"id integer, embedding vector({embeddings.shape[1]})"
        cur.execute(f"CREATE TABLE items ({table_definition})")
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        
        if labels is not None and label_names is not None:
            with cur.copy(f"COPY items (id, embedding, {', '.join(label_names)}) FROM STDIN WITH (FORMAT BINARY)") as copy:
                copy.set_types(["int4", "vector"] + ["int4" for _ in label_names])
                for i, embedding in enumerate(embeddings):
                    copy.write_row((i, embedding) + tuple(int(x) for x in labels[i]))
        else:
            with cur.copy("COPY items (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
                copy.set_types(["int4", "vector"])
                for i, embedding in enumerate(embeddings):
                    copy.write_row((i, embedding))
        
        print("Creating index...")
        
        if self._metric == "angular":
            cur.execute("CREATE INDEX ON items USING %s (embedding vector_cosine_ops) WITH (m = %d, ef_construction = %d)" % (self.index,self._m, self._ef_construction))
        elif self._metric == "euclidean":
            cur.execute("CREATE INDEX ON items USING %s (embedding vector_l2_ops) WITH (m = %d, ef_construction = %d)" % (self.index,self._m, self._ef_construction))
        else:
            raise RuntimeError(f"Unknown metric {self._metric}")
        
        print("Done!")
        self._cur = cur

    def parse_filter_expr(self, filter_expr: str) -> str:
        """Parse filter expression and return SQL WHERE clause"""

        print(f"Received filter expression: {filter_expr}")
        return filter_expr

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute("SET %s.ef_search = %d" % (self.index,ef_search))

    def query(self, v: np.array, n: int, filter_expr: str | None = None) -> list[int]:
        if filter_expr:
            filter_expr = filter_expr.replace("==", "=")
            query = f"""
                SELECT id FROM items 
                WHERE {filter_expr}
                ORDER BY embedding <-> %s 
                LIMIT %s
            """
        else:
            query = self._query
        self._cur.execute(query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self):
        if self._cur is None:
            return 0
        self._cur.execute("SELECT pg_relation_size('items_embedding_idx')")
        return self._cur.fetchone()[0] / 1024

    def __str__(self):
        return f"PGVector(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"
    
    def insert(self, embeddings: np.ndarray, labels: np.ndarray | None = None) -> None:
        """
        Single insert data

        Args:
            embeddings (np.ndarray): embeddings
            labels (np.ndarray): labels

        Returns:
            None
        """
        if labels is not None and self.label_names is not None:
            insert_sentence = (f"INSERT INTO items (id,embedding,{', '.join(self.label_names)}) VALUES ({self.num_entities+1},{embeddings},{', '.join(labels)})")
        else:
            insert_sentence = (f"INSERT INTO items (id,embedding) VALUES ({self.num_entities+1},{embeddings}")
        self._cur.execute(insert_sentence)
        self.num_entities += 1

    def update(
        self, index: int, embeddings: np.ndarray, labels: np.ndarray | None = None
    ) -> None:
        """
        Single update data

        Args:
            index (int): index to update
            embeddings (np.ndarray): embeddings
            labels (np.ndarray): labels

        Returns:
            None
        """
        update_item = (f"embeddings = {embeddings},")
        if labels is not None and self.label_names is not None:
            for i in enumerate(self.label_names):
                update_item += f"{self.label_names[i]} = {labels[i]}"
        update_sentence = (f"UPDATE items SET {update_item} where id = {index}")

        self._cur.execute(update_sentence)

    def delete(
        self,
        index: int,
    ) -> None:
        """
        Single delete data

        Args:
            index (int): index to delete

        Returns:
            None
        """
        delete_sentence = (f"DELETE FROM items where id = {index}")

        self._cur.execute(delete_sentence)


class PGVectorHNSW(PGVector):
    def __init__(self, metric: str, index_param: dict):
        super().__init__(metric, index_param)
        self._nlinks = index_param.get("nlinks", 32)
        self._efConstruction = index_param.get("efConstruction", 40)

    def get_vector_index(self):
        """Get HNSW vector index"""
        return "hnsw"

    def set_query_arguments(self, efSearch: int = 40):
        """
        Set query arguments for pgvector query with hnsw index
        """
        self.search_params = {
            "metric_type": self._metric_type,
            "efSearch": efSearch,
        }
        self.name = f"pgvector HNSW metric:{self._metric}, nlinks:{self._nlinks}, efConstruction:{self._efConstruction}, efSearch:{efSearch}"

class PGVectorIVFFLAT(PGVector):
    def __init__(self, metric: str, index_param: dict):
        super().__init__(metric, index_param)
        self._nlinks = index_param.get("nlinks", 32)
        self._efConstruction = index_param.get("efConstruction", 40)

    def get_vector_index(self):
        """Get IVFFLAT vector index"""
        return "ivfflat"

    def set_query_arguments(self, efSearch: int = 40):
        """
        Set query arguments for pgvector query with ivfflat index
        """
        self.search_params = {
            "metric_type": self._metric_type,
            "efSearch": efSearch,
        }
        self.name = f"pgvector ivfflat metric:{self._metric}, nlinks:{self._nlinks}, efConstruction:{self._efConstruction}, efSearch:{efSearch}"