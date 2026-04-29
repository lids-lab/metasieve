from typing import Tuple, List, Protocol

EdgeType = Tuple[str, str, str]
MetaPath = List[EdgeType]


def _prefix_key(prefix: MetaPath) -> str:
    if not prefix:
        raise ValueError("prefix cannot be empty (hop-0 has no frontier table)")
    parts = [prefix[0][0]]
    for (_, rel, dst) in prefix:
        parts.append(rel)
        parts.append(dst)
    return "__".join(parts)


def frontier_name(prefix: MetaPath) -> str:
    return f"frontiers.frontier__{_prefix_key(prefix)}"


def _hop_suffix(prefix: MetaPath) -> Tuple[str, int]:
    hop = len(prefix)
    last_dst = prefix[-1][2]
    return last_dst, hop


class FeatureStrategy(Protocol):
    feature_key: str
    name: str

    def feature_table_name(self, prefix: MetaPath) -> str: ...
    def build_sql(self, prefix: MetaPath) -> str: ...
    def export_column_name(self, prefix: MetaPath) -> str: ...


class CountDistinctStrategy:
    feature_key = "cnt"
    name = "count_distinct"

    def __init__(self, seed_id_col: str = "SeedId"):
        self.seed_id_col = seed_id_col

    def feature_table_name(self, prefix: MetaPath) -> str:
        last_dst, hop = _hop_suffix(prefix)
        return f"features.{self.feature_key}__{_prefix_key(prefix)}__{last_dst}_h{hop}"

    def export_column_name(self, prefix: MetaPath) -> str:
        last_dst, hop = _hop_suffix(prefix)
        return f"cnt_{last_dst}_h{hop}"

    def build_sql(self, prefix: MetaPath) -> str:
        front = frontier_name(prefix)
        col_name = self.export_column_name(prefix)
        return f"""
        CREATE OR REPLACE TABLE {self.feature_table_name(prefix)} AS
        SELECT
          {self.seed_id_col},
          timestamp,
          COUNT(DISTINCT last_id) AS {col_name}
        FROM {front}
        GROUP BY {self.seed_id_col}, timestamp
        """


class LogCountStrategy:
    feature_key = "logcnt"
    name = "log_count" 

    def __init__(self, seed_id_col: str = "SeedId"):
        self.seed_id_col = seed_id_col

    def feature_table_name(self, prefix: MetaPath) -> str:
        last_dst, hop = _hop_suffix(prefix)
        return f"features.{self.feature_key}__{_prefix_key(prefix)}__{last_dst}_h{hop}"

    def export_column_name(self, prefix: MetaPath) -> str:
        last_dst, hop = _hop_suffix(prefix)
        return f"logcnt_{last_dst}_h{hop}"

    def build_sql(self, prefix: MetaPath) -> str:
        front = frontier_name(prefix)
        col_name = self.export_column_name(prefix)
        return f"""
        CREATE OR REPLACE TABLE {self.feature_table_name(prefix)} AS
        SELECT
          {self.seed_id_col},
          timestamp,
          CAST(LN(1 + COUNT(DISTINCT last_id)) AS DOUBLE) AS {col_name}
        FROM {front}
        GROUP BY {self.seed_id_col}, timestamp
        """


class RateStrategy:
    feature_key = "rate"
    name = "rate"

    def __init__(self, seed_id_col: str = "SeedId"):
        self.seed_id_col = seed_id_col

    def feature_table_name(self, prefix: MetaPath) -> str:
        last_dst, hop = _hop_suffix(prefix)
        return f"features.{self.feature_key}__{_prefix_key(prefix)}__{last_dst}_h{hop}"

    def export_column_name(self, prefix: MetaPath) -> str:
        last_dst, hop = _hop_suffix(prefix)
        return f"rate_{last_dst}_h{hop}"

    def build_sql(self, prefix: MetaPath) -> str:
        hop = len(prefix)
        child_front = frontier_name(prefix)
        last_dst, hop_idx = _hop_suffix(prefix)
        out_table = self.feature_table_name(prefix)
        out_col = self.export_column_name(prefix)

        if hop_idx == 1:
            return f"""
            CREATE OR REPLACE TABLE {out_table} AS
            WITH child AS (
              SELECT
                {self.seed_id_col},
                timestamp,
                COUNT(DISTINCT last_id) AS num_cnt
              FROM {child_front}
              GROUP BY {self.seed_id_col}, timestamp
            )
            SELECT
              {self.seed_id_col},
              timestamp,
              CASE
                WHEN num_cnt IS NULL THEN 0.0
                ELSE CAST(num_cnt AS DOUBLE) / 1.0
              END AS {out_col}
            FROM child
            """

        parent_prefix = prefix[:-1]
        parent_front = frontier_name(parent_prefix)

        num_alias = f"num_{hop_idx}"
        den_alias = f"den_{hop_idx}"

        return f"""
        CREATE OR REPLACE TABLE {out_table} AS
        WITH child AS (
          SELECT
            {self.seed_id_col},
            timestamp,
            COUNT(DISTINCT last_id) AS {num_alias}
          FROM {child_front}
          GROUP BY {self.seed_id_col}, timestamp
        ),
        parent AS (
          SELECT
            {self.seed_id_col},
            timestamp,
            COUNT(DISTINCT last_id) AS {den_alias}
          FROM {parent_front}
          GROUP BY {self.seed_id_col}, timestamp
        )
        SELECT
          COALESCE(c.{self.seed_id_col}, p.{self.seed_id_col}) AS {self.seed_id_col},
          COALESCE(c.timestamp, p.timestamp)                   AS timestamp,
          CASE
            WHEN {den_alias} IS NULL OR {den_alias} = 0 THEN 0.0
            ELSE CAST({num_alias} AS DOUBLE) / CAST({den_alias} AS DOUBLE)
          END AS {out_col}
        FROM child c
        FULL OUTER JOIN parent p
          ON c.{self.seed_id_col} = p.{self.seed_id_col}
         AND c.timestamp = p.timestamp
        """


class LogRateStrategy:
    feature_key = "lograte"
    name = "log_rate" 

    def __init__(self, seed_id_col: str = "SeedId"):
        self.seed_id_col = seed_id_col

    def feature_table_name(self, prefix: MetaPath) -> str:
        last_dst, hop = _hop_suffix(prefix)
        return f"features.{self.feature_key}__{_prefix_key(prefix)}__{last_dst}_h{hop}"

    def export_column_name(self, prefix: MetaPath) -> str:
        last_dst, hop = _hop_suffix(prefix)
        return f"lograte_{last_dst}_h{hop}"

    def build_sql(self, prefix: MetaPath) -> str:
        child_front = frontier_name(prefix)
        last_dst, hop_idx = _hop_suffix(prefix)
        out_table = self.feature_table_name(prefix)
        out_col = self.export_column_name(prefix)

        if hop_idx == 1:
            return f"""
            CREATE OR REPLACE TABLE {out_table} AS
            WITH child AS (
              SELECT
                {self.seed_id_col},
                timestamp,
                COUNT(DISTINCT last_id) AS num_cnt
              FROM {child_front}
              GROUP BY {self.seed_id_col}, timestamp
            )
            SELECT
              {self.seed_id_col},
              timestamp,
              CASE
                WHEN num_cnt IS NULL THEN 0.0
                ELSE CAST(LN(1 + (CAST(num_cnt AS DOUBLE) / 1.0)) AS DOUBLE)
              END AS {out_col}
            FROM child
            """

        parent_prefix = prefix[:-1]
        parent_front = frontier_name(parent_prefix)

        num_alias = f"num_{hop_idx}"
        den_alias = f"den_{hop_idx}"

        return f"""
        CREATE OR REPLACE TABLE {out_table} AS
        WITH child AS (
          SELECT
            {self.seed_id_col},
            timestamp,
            COUNT(DISTINCT last_id) AS {num_alias}
          FROM {child_front}
          GROUP BY {self.seed_id_col}, timestamp
        ),
        parent AS (
          SELECT
            {self.seed_id_col},
            timestamp,
            COUNT(DISTINCT last_id) AS {den_alias}
          FROM {parent_front}
          GROUP BY {self.seed_id_col}, timestamp
        )
        SELECT
          COALESCE(c.{self.seed_id_col}, p.{self.seed_id_col}) AS {self.seed_id_col},
          COALESCE(c.timestamp, p.timestamp)                   AS timestamp,
          CASE
            WHEN {den_alias} IS NULL OR {den_alias} = 0 THEN 0.0
            ELSE CAST(
              LN(1 + (CAST({num_alias} AS DOUBLE) / CAST({den_alias} AS DOUBLE)))
              AS DOUBLE
            )
          END AS {out_col}
        FROM child c
        FULL OUTER JOIN parent p
          ON c.{self.seed_id_col} = p.{self.seed_id_col}
         AND c.timestamp = p.timestamp
        """
