"""Compatibility boundary for the course SDK and the local RAGFlow service."""
import requests
from ragflow_sdk import RAGFlow
from ragflow_sdk.modules.chat import Chat


class CompatibleRAGFlow(RAGFlow):
    def _request(self, method, path, **kwargs):
        response = requests.request(
            method, self.api_url + path, headers=self.authorization_header,
            timeout=(5, 90), **kwargs
        )
        response.raise_for_status()
        return response

    def get(self, path, params=None, json=None):
        return self._request("GET", path, params=params, json=json)

    def post(self, path, json=None, stream=False, files=None):
        return self._request("POST", path, json=json, stream=stream, files=files)

    def put(self, path, json):
        return self._request("PUT", path, json=json)

    def delete(self, path, json):
        return self._request("DELETE", path, json=json)

    def list_chats(self, page=1, page_size=30, orderby="create_time", desc=True, id=None, name=None):
        payload = self.get("/chats", params={
            "page": page, "page_size": page_size, "orderby": orderby,
            "desc": desc, "id": id, "name": name,
        }).json()
        if payload.get("code") != 0:
            raise ValueError(payload.get("message", "RAGFlow list_chats failed"))
        data = payload.get("data", [])
        # 0.24 返回 data: list；0.27.1 返回 data: {chats: list, total: int}。
        if isinstance(data, dict):
            data = data.get("chats", [])
        if not isinstance(data, list):
            raise ValueError("Unexpected RAGFlow chats response shape")
        return [Chat(self, item) for item in data]
