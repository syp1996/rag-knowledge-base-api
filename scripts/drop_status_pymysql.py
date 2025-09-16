#!/usr/bin/env python3
import os
def try_load_env():
    # try python-dotenv first
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except Exception:
        pass
    # simple fallback parser
    try:
        path = os.path.join(os.getcwd(), '.env')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line=line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        k, v = line.split('=', 1)
                        os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass
import pymysql

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
    try:
        cur.execute('DROP INDEX idx_user_id_status ON documents')
        print('Dropped index idx_user_id_status')
    except Exception as e:
        print('Skip drop index:', e)
    try:
        cur.execute('ALTER TABLE documents DROP COLUMN status')
        print('Dropped column status')
    except Exception as e:
        print('Skip drop column:', e)
    # Also remove soft-deleted rows and drop deleted_at column
    try:
        cur.execute('DELETE FROM documents WHERE deleted_at IS NOT NULL')
        print('Deleted previously soft-deleted documents')
    except Exception as e:
        print('Skip deleting soft-deleted rows:', e)
    try:
        cur.execute('ALTER TABLE documents DROP COLUMN deleted_at')
        print('Dropped column deleted_at')
    except Exception as e:
        print('Skip drop deleted_at:', e)
    cur.close(); conn.close()
    print('Done.')

if __name__ == '__main__':
    main()
