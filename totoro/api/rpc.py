#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   rpc.py
@Time    :   2024/11/08 15:45:49
@Desc    :   
'''
import grpc
from concurrent import futures
from totoro.services.service import RAGService
from totoro.pb import services_pb2_grpc


def grpc_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    services_pb2_grpc.add_RAGCoreServiceServicer_to_server(
        RAGService(), server)
    server.add_insecure_port('[::]:50051')  # gRPC server runs on port 50051
    server.start()


if __name__ == '__main__':
    grpc_server()
