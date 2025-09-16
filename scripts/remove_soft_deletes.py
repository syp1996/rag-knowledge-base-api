#!/usr/bin/env python3
import os
import pymysql

def try_load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

def main():
    try_load_env()
    cfg = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '3306')),
        'db': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'charset': 'utf8mb4',
        'autocommit': True,
    }
    print('Connecting to MySQL...', cfg['host'])
    conn = pymysql.connect(**cfg)
    cur = conn.cursor()
    # 1) 物理删除曾经软删除的文档行（CASCADE 自动清理 doc_chunks）
    try:
        cur.execute('DELETE FROM documents WHERE deleted_at IS NOT NULL')
        print('Deleted previously soft-deleted documents')
    except Exception as e:
        print('Skip deleting soft-deleted rows (maybe column missing):', e)
    # 2) 删除 deleted_at 列
    try:
        cur.execute('ALTER TABLE documents DROP COLUMN deleted_at')
        print('Dropped column deleted_at')
    except Exception as e:
        print('Skip drop deleted_at (maybe already dropped):', e)
    cur.close(); conn.close()
    print('Done.')

if __name__ == '__main__':
    main()

