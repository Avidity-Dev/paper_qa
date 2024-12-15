from flask import request, jsonify
from flask.views import MethodView


class ChatAPI(MethodView):
    """"""
    def post(self):
        data = request.get_json()
        return jsonify({"message": "Hello, World!"})


class DocumentAPI(MethodView):
