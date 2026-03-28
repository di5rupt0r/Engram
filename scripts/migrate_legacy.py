#!/usr/bin/env python3
"""
Migration script to backfill data from legacy basic-memory SQLite to Engram Redis.
Place this in scripts/migrate_legacy.py and run it from the root of the repo.
"""

import sqlite3
import os
import sys
import logging
from datetime import datetime
from typing import Optional

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engram.redis.client import EngramRedisClient
from engram.embeddings.provider import generate_embedding

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def migrate(sqlite_path: str, redis_host: str = "localhost", redis_port: int = 6379):
    if not os.path.exists(sqlite_path):
        logger.error(f"SQLite file not found: {sqlite_path}")
        return

    logger.info(f"Connecting to legacy DB: {sqlite_path}")
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    logger.info(f"Connecting to Engram Redis at {redis_host}:{redis_port}")
    redis_client = EngramRedisClient(host=redis_host, port=redis_port)

    # 1. Migrate Entities
    logger.info("Migrating entities...")
    cursor.execute("SELECT * FROM entity")
    entities = cursor.fetchall()
    for row in entities:
        node_id = f"legacy_entity_{row['id']}"
        content = f"{row['title']}\n{row['file_path']}"
        domain = "legacy_basic_memory"
        node_type = row['note_type'] or "entity"
        
        try:
            embedding = generate_embedding(content)
            metadata = {
                "legacy_id": row['id'],
                "file_path": row['file_path'],
                "created_at": row['created_at']
            }
            
            success = redis_client.store_node(
                node_id=node_id,
                domain=domain,
                node_type=node_type,
                content=content,
                embedding=embedding,
                metadata=metadata
            )
            
            if success:
                redis_client.update_manifest(domain, node_id)
                logger.info(f"Migrated entity: {row['title']}")
        except Exception as e:
            logger.error(f"Failed to migrate entity {row['id']}: {e}")

    # 2. Migrate Observations
    logger.info("Migrating observations...")
    cursor.execute("SELECT * FROM observation")
    observations = cursor.fetchall()
    for row in observations:
        node_id = f"legacy_obs_{row['id']}"
        content = row['content']
        domain = "legacy_basic_memory"
        node_type = "observation"
        
        try:
            embedding = generate_embedding(content)
            metadata = {
                "legacy_id": row['id'],
                "entity_id": f"legacy_entity_{row['entity_id']}",
                "category": row['category']
            }
            
            success = redis_client.store_node(
                node_id=node_id,
                domain=domain,
                node_type=node_type,
                content=content,
                embedding=embedding,
                metadata=metadata
            )
            
            if success:
                redis_client.update_manifest(domain, node_id)
                logger.info(f"Migrated observation: {row['id']}")
        except Exception as e:
            logger.error(f"Failed to migrate observation {row['id']}: {e}")

    conn.close()
    logger.info("Migration complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Migrate legacy basic-memory data to Engram.")
    parser.add_argument("--db", default="/home/gabriel/.basic-memory/memory.db", help="Path to legacy SQLite DB")
    parser.add_argument("--host", default="localhost", help="Redis host")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    
    args = parser.parse_args()
    migrate(args.db, args.host, args.port)
