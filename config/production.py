# coding=utf-8

"""生产环境使用的配置
"""

DB_DEBUG = True
DB = 'ixx_db'
HOST = '192.168.1.99'
USER = 'ixianxia'
PASSWD = 'eojj2oHu4pKQLGgD'
CONNECT_STRING = 'mysql://%s:%s@%s/%s?charset=utf8' % (USER, PASSWD, HOST, DB)
