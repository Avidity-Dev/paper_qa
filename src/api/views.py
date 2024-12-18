from flask import request, jsonify
from flask.views import MethodView
from paperqa.settings import Settings as PQASettings

from src.config.config import PQASettings
from src.storage.vector.stores import RedisVectorStore


class ChatAPI(MethodView):
    """Verbs associated with chatting with the document store."""

    def __init__(self, pqa_settings: PQASettings, vector_store: RedisVectorStore):
        self.pqa_settings = pqa_settings
        self.vector_store = vector_store

    def post(self):
        data = request.get_json()
        return jsonify({"message": "Hello, World!"})


class DocumentAPI(MethodView):
    """Verbs associated with uploading and retrieving documents."""

    def post(self):
        data = request.get_json()
        return jsonify({"message": "Hello, World!"})
