"""
Commands for application environment management.
"""

import os

import click
import redis

from src.config.config import VEC_IDX_NAME, VEC_IDX_PREFIX

FT_CREATE_COMMAND = f"""
    FT.CREATE {VEC_IDX_NAME} ON JSON
        PREFIX 1 {VEC_IDX_PREFIX} SCORE 1.0
        SCHEMA
"""
