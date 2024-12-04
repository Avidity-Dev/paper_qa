"""
Commands for application environment management.
"""

import os

import click
import redis

from src.config.config import VEC_IDX_NAME, VEC_IDX_PREFIX

# TODO: Create a function to generate this command from config and be able to run from
# from the command line.
FT_CREATE_COMMAND = """
FT.CREATE idx:docs_vss ON JSON PREFIX 1 docs: SCHEMA $.id AS id TEXT NOSTEM $.title AS title TEXT WEIGHT 1.0 $.authors AS authors TAG SEPARATOR "|" $.doi AS doi TEXT NOSTEM $.published_date AS published_date NUMERIC SORTABLE $.created_at AS created_at NUMERIC SORTABLE $.content_chunks[*].text AS chunk_text TEXT WEIGHT 1.0 $.content_chunks[*].embedding AS vector VECTOR FLAT 6 TYPE FLOAT32 DIM 1536 DISTANCE_METRIC COSINE
"""
