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
from __future__ import annotations

from typing import Optional, Sequence, Union

from openai._types import NOT_GIVEN, NotGiven

from camel.configs.base_config import BaseConfig


class MiniMaxConfig(BaseConfig):
    r"""Defines the parameters for generating chat completions using the
    MiniMax API (OpenAI-compatible).

    MiniMax requires temperature in (0.0, 1.0]. Values outside this range
    are clamped automatically.

    Args:
        temperature (float, optional): Sampling temperature, clamped to
            (0.0, 1.0] for MiniMax. (default: :obj:`0.2`)
        top_p (float, optional): Nucleus sampling parameter.
            (default: :obj:`1.0`)
        max_tokens (int, optional): Maximum number of tokens to generate.
            (default: :obj:`NOT_GIVEN`)
        stream (bool, optional): Whether to stream partial results.
            (default: :obj:`False`)
        stop (str or list, optional): Stop sequences.
            (default: :obj:`NOT_GIVEN`)
    """

    temperature: float = 0.2
    top_p: float = 1.0
    stream: bool = False
    stop: Union[str, Sequence[str], NotGiven] = NOT_GIVEN
    max_tokens: Union[int, NotGiven] = NOT_GIVEN


MINIMAX_API_PARAMS = {param for param in MiniMaxConfig.model_fields.keys()}
