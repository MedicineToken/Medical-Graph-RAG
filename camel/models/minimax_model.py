# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import os
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI, Stream

from camel.configs import MINIMAX_API_PARAMS
from camel.messages import OpenAIMessage
from camel.models import BaseModelBackend
from camel.types import ChatCompletion, ChatCompletionChunk, ModelType
from camel.utils import (
    BaseTokenCounter,
    OpenAITokenCounter,
    api_keys_required,
)


class MiniMaxModel(BaseModelBackend):
    r"""MiniMax API in a unified BaseModelBackend interface.

    Uses MiniMax's OpenAI-compatible endpoint at https://api.minimax.io/v1.
    Supports MiniMax-M2.7 and MiniMax-M2.7-highspeed models (204K context).
    Temperature is automatically clamped to (0.0, 1.0].
    """

    def __init__(
        self,
        model_type: ModelType,
        model_config_dict: Dict[str, Any],
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
    ) -> None:
        super().__init__(
            model_type, model_config_dict, api_key, url, token_counter
        )
        self._url = url or os.environ.get(
            "MINIMAX_API_BASE_URL", "https://api.minimax.io/v1"
        )
        self._api_key = api_key or os.environ.get("MINIMAX_API_KEY")
        self._client = OpenAI(
            timeout=60,
            max_retries=3,
            base_url=self._url,
            api_key=self._api_key,
        )

    @property
    def token_counter(self) -> BaseTokenCounter:
        if not self._token_counter:
            self._token_counter = OpenAITokenCounter(ModelType.GPT_4O)
        return self._token_counter

    @api_keys_required("MINIMAX_API_KEY")
    def run(
        self,
        messages: List[OpenAIMessage],
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        r"""Runs inference of MiniMax chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.

        Returns:
            Union[ChatCompletion, Stream[ChatCompletionChunk]]:
                `ChatCompletion` in the non-stream mode, or
                `Stream[ChatCompletionChunk]` in the stream mode.
        """
        config = dict(self.model_config_dict)
        # Clamp temperature to MiniMax's valid range (0.0, 1.0]
        if "temperature" in config:
            config["temperature"] = max(0.01, min(config["temperature"], 1.0))

        response = self._client.chat.completions.create(
            messages=messages,
            model=self.model_type.value,
            **config,
        )
        return response

    def check_model_config(self):
        r"""Check whether the model configuration contains any
        unexpected arguments to MiniMax API.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments.
        """
        for param in self.model_config_dict:
            if param not in MINIMAX_API_PARAMS:
                raise ValueError(
                    f"Unexpected argument `{param}` is "
                    "input into MiniMax model backend."
                )

    @property
    def stream(self) -> bool:
        return self.model_config_dict.get('stream', False)
