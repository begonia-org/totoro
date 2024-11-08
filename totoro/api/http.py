#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   http.py
@Time    :   2024/11/08 17:01:11
@Desc    :   
'''
import uvicorn
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel

from totoro.services.service import RAGService

from totoro.models import services_model


class BaseHttpResponse(BaseModel):
    code: int = 0
    message: str = "success"
    data: BaseModel


class EmbeddingHttpResponse(BaseHttpResponse):
    data: services_model.EmbeddingResponse


class RerankingHttpResponse(BaseHttpResponse):
    data: services_model.ReankingResponse


class QueryHttpResponse(BaseHttpResponse):
    data: services_model.QueryBuildResponse


class HTTPServer:
    def __init__(self):
        self.service = RAGService()
        self.app = FastAPI()
        self.rag_router = APIRouter(prefix="/api/v1/rag")
        self.rag_router.add_api_route(
            "/embedding", self.embedding, methods=["POST"])
        self.rag_router.add_api_route(
            "/reranking", self.reranking, methods=["POST"])
        self.rag_router.add_api_route(
            "/query", self.query, methods=["POST"])
        self.app.include_router(self.rag_router)

    def start(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000)

    def embedding(self, req: services_model.EmbeddingRequest) -> EmbeddingHttpResponse:
        """
        Embedding
        """

        rsp = self.service.embedding(req.to_protobuf(), None)
        return EmbeddingHttpResponse(data=services_model.EmbeddingResponse.from_protobuf(rsp))

    def reranking(self, req: services_model.ReankingRequest) -> RerankingHttpResponse:
        """
        Re-ranking
        """
        rsp = self.service.reanking(req.to_protobuf(), None)
        return RerankingHttpResponse(data=services_model.ReankingResponse.from_protobuf(rsp))

    def query(self, req: services_model.QueryBuildRequest) -> BaseHttpResponse:
        """
        Query
        """
        rsp = self.service.query(req.to_protobuf(), None)
        return BaseHttpResponse(data=services_model.QueryBuildResponse.from_protobuf(rsp))
