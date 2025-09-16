#!/usr/bin/env python3
import os
import mysql.connector

def main():
    cfg = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '3306')),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
    }
    print('Connecting to MySQL...', cfg['host'])
    cn = mysql.connector.connect(**cfg)
    cur = cn.cursor()
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
    cn.commit()
    cur.close(); cn.close()
    print('Done.')

if __name__ == '__main__':
    main()

