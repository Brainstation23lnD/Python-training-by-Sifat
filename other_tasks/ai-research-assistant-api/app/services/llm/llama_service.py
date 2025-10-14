import json
import requests
from langchain.llms.base import LLM
from typing import Optional, List


class LLaMAWrapper(LLM):
    endpoint: str  # ✅ must have a type annotation

    @property
    def _llm_type(self) -> str:
        return "llama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = requests.post(
            self.endpoint,
            json={"model": "llama2", "prompt": prompt, "max_tokens": 512},

        )

        output_chunks = []
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    output_chunks.append(data["response"])
                if data.get("done"):
                    break
            except json.JSONDecodeError:
                continue

        return "".join(output_chunks)
