#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   rdb.py
@Time    :   2024/11/13 14:41:31
@Desc    :   
'''


import threading

import redis


class RDB:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self, host='localhost', port=6379, db=0, password=""):
        self.connection_pool = redis.ConnectionPool(
            host=host, port=port, db=db, password=password, decode_responses=True)
        self.redis_instance = redis.Redis(
            connection_pool=self.connection_pool, decode_responses=True)

    def get(self, key):
        return self.redis_instance.get(key)

    def set(self, key, value, ex):
        self.redis_instance.set(key, value, ex)

    def get_rdb(self) -> redis.Redis:
        return self.redis_instance

    def push(self, key, value):
        self.redis_instance.rpush(key, value)
