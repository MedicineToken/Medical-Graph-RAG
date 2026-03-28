"""Integration tests for MiniMax LLM provider.

These tests require a valid MINIMAX_API_KEY environment variable and
make real API calls to the MiniMax service. They are skipped when
the key is not available.
"""
import os
import unittest
import asyncio

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
SKIP_REASON = "MINIMAX_API_KEY not set"


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxUtilsIntegration(unittest.TestCase):
    """Integration test for utils.py call_llm with MiniMax."""

    def test_call_llm_with_minimax(self):
        """Test end-to-end LLM call via MiniMax provider."""
        os.environ["LLM_PROVIDER"] = "minimax"
        from utils import call_llm
        result = call_llm(
            "You are a helpful assistant.",
            "What is 2+2? Answer with just the number."
        )
        self.assertIsInstance(result, str)
        self.assertIn("4", result)


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxNanoGraphRAGIntegration(unittest.TestCase):
    """Integration test for nano_graphrag MiniMax functions."""

    def test_minimax_m27_complete(self):
        """Test MiniMax M2.7 completion via nano_graphrag."""
        from nano_graphrag._llm import minimax_m27_complete

        async def _run():
            result = await minimax_m27_complete(
                "What is 1+1? Answer with just the number."
            )
            return result

        result = asyncio.get_event_loop().run_until_complete(_run())
        self.assertIsInstance(result, str)
        self.assertIn("2", result)

    def test_minimax_m27_highspeed_complete(self):
        """Test MiniMax M2.7-highspeed completion."""
        from nano_graphrag._llm import minimax_m27_highspeed_complete

        async def _run():
            result = await minimax_m27_highspeed_complete(
                "What is 3+3? Answer with just the number."
            )
            return result

        result = asyncio.get_event_loop().run_until_complete(_run())
        self.assertIsInstance(result, str)
        self.assertIn("6", result)


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxModelIntegration(unittest.TestCase):
    """Integration test for CAMEL MiniMax model backend."""

    def test_camel_minimax_model_run(self):
        """Test CAMEL ModelFactory-created MiniMax model."""
        from camel.models import ModelFactory
        from camel.types import ModelPlatformType, ModelType

        model = ModelFactory.create(
            model_platform=ModelPlatformType.MINIMAX,
            model_type=ModelType.MINIMAX_M27,
            model_config_dict={"temperature": 0.5, "max_tokens": 50},
        )
        messages = [
            {"role": "user", "content": "What is 5+5? Answer with just the number."}
        ]
        response = model.run(messages)
        content = response.choices[0].message.content
        self.assertIsInstance(content, str)
        self.assertIn("10", content)


if __name__ == "__main__":
    unittest.main()
