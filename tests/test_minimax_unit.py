"""Unit tests for MiniMax LLM provider integration."""
import os
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio


class TestProviderDetection(unittest.TestCase):
    """Test provider auto-detection logic in utils.py."""

    def _import_utils(self):
        """Import utils module functions directly."""
        import importlib
        import sys
        # We need to handle the import chain carefully
        # utils.py imports from summerize.py, so we mock that chain
        if 'utils' in sys.modules:
            importlib.reload(sys.modules['utils'])
        # Import the detection functions from utils
        from utils import _detect_provider, PROVIDER_PRESETS, _clamp_temperature
        return _detect_provider, PROVIDER_PRESETS, _clamp_temperature

    @patch.dict(os.environ, {}, clear=True)
    def test_default_provider_is_openai(self):
        """When no env vars are set, default to openai."""
        _detect_provider, _, _ = self._import_utils()
        self.assertEqual(_detect_provider(), "openai")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}, clear=True)
    def test_auto_detect_minimax_from_api_key(self):
        """MINIMAX_API_KEY triggers minimax auto-detection."""
        _detect_provider, _, _ = self._import_utils()
        self.assertEqual(_detect_provider(), "minimax")

    @patch.dict(os.environ, {"LLM_PROVIDER": "minimax"}, clear=True)
    def test_explicit_provider_env_var(self):
        """LLM_PROVIDER env var takes priority."""
        _detect_provider, _, _ = self._import_utils()
        self.assertEqual(_detect_provider(), "minimax")

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "openai",
        "MINIMAX_API_KEY": "test-key"
    }, clear=True)
    def test_explicit_provider_overrides_auto_detect(self):
        """LLM_PROVIDER=openai overrides MINIMAX_API_KEY auto-detection."""
        _detect_provider, _, _ = self._import_utils()
        self.assertEqual(_detect_provider(), "openai")

    def test_provider_presets_have_required_keys(self):
        """Verify all provider presets have required configuration keys."""
        _, PROVIDER_PRESETS, _ = self._import_utils()
        for name, preset in PROVIDER_PRESETS.items():
            self.assertIn("api_key_env", preset, f"{name} missing api_key_env")
            self.assertIn("default_model", preset, f"{name} missing default_model")

    def test_minimax_preset_values(self):
        """Verify MiniMax preset has correct defaults."""
        _, PROVIDER_PRESETS, _ = self._import_utils()
        mm = PROVIDER_PRESETS["minimax"]
        self.assertEqual(mm["api_key_env"], "MINIMAX_API_KEY")
        self.assertEqual(mm["default_base_url"], "https://api.minimax.io/v1")
        self.assertEqual(mm["default_model"], "MiniMax-M2.7")
        self.assertIsNone(mm["embedding_model"])


class TestTemperatureClamping(unittest.TestCase):
    """Test temperature clamping for MiniMax."""

    def _import_clamp(self):
        from utils import _clamp_temperature
        return _clamp_temperature

    def test_minimax_clamp_zero(self):
        """MiniMax temp=0 should clamp to 0.01."""
        clamp = self._import_clamp()
        self.assertAlmostEqual(clamp(0.0, "minimax"), 0.01)

    def test_minimax_clamp_negative(self):
        """MiniMax negative temp should clamp to 0.01."""
        clamp = self._import_clamp()
        self.assertAlmostEqual(clamp(-1.0, "minimax"), 0.01)

    def test_minimax_clamp_above_one(self):
        """MiniMax temp>1.0 should clamp to 1.0."""
        clamp = self._import_clamp()
        self.assertAlmostEqual(clamp(1.5, "minimax"), 1.0)

    def test_minimax_valid_temp_unchanged(self):
        """MiniMax temp in valid range should stay unchanged."""
        clamp = self._import_clamp()
        self.assertAlmostEqual(clamp(0.5, "minimax"), 0.5)

    def test_minimax_temp_one_unchanged(self):
        """MiniMax temp=1.0 is the upper bound, should be unchanged."""
        clamp = self._import_clamp()
        self.assertAlmostEqual(clamp(1.0, "minimax"), 1.0)

    def test_openai_no_clamping(self):
        """OpenAI provider should not clamp temperature."""
        clamp = self._import_clamp()
        self.assertAlmostEqual(clamp(1.5, "openai"), 1.5)
        self.assertAlmostEqual(clamp(0.0, "openai"), 0.0)


class TestSummerizeProviderDetection(unittest.TestCase):
    """Test provider detection in summerize.py."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}, clear=True)
    def test_summerize_detects_minimax(self):
        from summerize import _detect_provider
        self.assertEqual(_detect_provider(), "minimax")

    @patch.dict(os.environ, {}, clear=True)
    def test_summerize_defaults_openai(self):
        from summerize import _detect_provider
        self.assertEqual(_detect_provider(), "openai")


class TestMiniMaxModelType(unittest.TestCase):
    """Test MiniMax model type enum values."""

    def test_minimax_model_types_exist(self):
        from camel.types import ModelType
        self.assertEqual(ModelType.MINIMAX_M27.value, "MiniMax-M2.7")
        self.assertEqual(ModelType.MINIMAX_M27_HIGHSPEED.value, "MiniMax-M2.7-highspeed")

    def test_minimax_is_minimax(self):
        from camel.types import ModelType
        self.assertTrue(ModelType.MINIMAX_M27.is_minimax)
        self.assertTrue(ModelType.MINIMAX_M27_HIGHSPEED.is_minimax)

    def test_openai_is_not_minimax(self):
        from camel.types import ModelType
        self.assertFalse(ModelType.GPT_4O.is_minimax)

    def test_minimax_is_not_openai(self):
        from camel.types import ModelType
        self.assertFalse(ModelType.MINIMAX_M27.is_openai)

    def test_minimax_token_limit(self):
        from camel.types import ModelType
        self.assertEqual(ModelType.MINIMAX_M27.token_limit, 204_000)
        self.assertEqual(ModelType.MINIMAX_M27_HIGHSPEED.token_limit, 204_000)


class TestMiniMaxPlatformType(unittest.TestCase):
    """Test MiniMax platform type enum."""

    def test_minimax_platform_exists(self):
        from camel.types import ModelPlatformType
        self.assertEqual(ModelPlatformType.MINIMAX.value, "minimax")

    def test_is_minimax(self):
        from camel.types import ModelPlatformType
        self.assertTrue(ModelPlatformType.MINIMAX.is_minimax)
        self.assertFalse(ModelPlatformType.OPENAI.is_minimax)


class TestMiniMaxConfig(unittest.TestCase):
    """Test MiniMax configuration class."""

    def test_config_defaults(self):
        from camel.configs import MiniMaxConfig
        cfg = MiniMaxConfig()
        self.assertEqual(cfg.temperature, 0.2)
        self.assertEqual(cfg.top_p, 1.0)
        self.assertFalse(cfg.stream)

    def test_config_api_params(self):
        from camel.configs import MINIMAX_API_PARAMS
        self.assertIn("temperature", MINIMAX_API_PARAMS)
        self.assertIn("top_p", MINIMAX_API_PARAMS)
        self.assertIn("max_tokens", MINIMAX_API_PARAMS)
        self.assertIn("stream", MINIMAX_API_PARAMS)
        self.assertIn("stop", MINIMAX_API_PARAMS)


class TestMiniMaxModel(unittest.TestCase):
    """Test MiniMax model backend."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_model_initialization(self):
        from camel.models.minimax_model import MiniMaxModel
        from camel.types import ModelType
        model = MiniMaxModel(
            model_type=ModelType.MINIMAX_M27,
            model_config_dict={"temperature": 0.5},
        )
        self.assertEqual(model._url, "https://api.minimax.io/v1")
        self.assertEqual(model._api_key, "test-key")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_model_custom_url(self):
        from camel.models.minimax_model import MiniMaxModel
        from camel.types import ModelType
        model = MiniMaxModel(
            model_type=ModelType.MINIMAX_M27,
            model_config_dict={},
            url="https://custom.minimax.io/v1",
        )
        self.assertEqual(model._url, "https://custom.minimax.io/v1")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_check_model_config_valid(self):
        from camel.models.minimax_model import MiniMaxModel
        from camel.types import ModelType
        model = MiniMaxModel(
            model_type=ModelType.MINIMAX_M27,
            model_config_dict={"temperature": 0.5, "max_tokens": 100},
        )
        model.check_model_config()  # Should not raise

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_check_model_config_invalid(self):
        from camel.models.minimax_model import MiniMaxModel
        from camel.types import ModelType
        with self.assertRaises(ValueError):
            MiniMaxModel(
                model_type=ModelType.MINIMAX_M27,
                model_config_dict={"invalid_param": True},
            )

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_stream_property(self):
        from camel.models.minimax_model import MiniMaxModel
        from camel.types import ModelType
        model = MiniMaxModel(
            model_type=ModelType.MINIMAX_M27,
            model_config_dict={"stream": True},
        )
        self.assertTrue(model.stream)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_temperature_clamping_in_run(self):
        """Verify that run() clamps temperature before sending to API."""
        from camel.models.minimax_model import MiniMaxModel
        from camel.types import ModelType
        model = MiniMaxModel(
            model_type=ModelType.MINIMAX_M27,
            model_config_dict={"temperature": 0.0},
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        model._client.chat.completions.create = MagicMock(return_value=mock_response)
        model.run([{"role": "user", "content": "test"}])
        call_kwargs = model._client.chat.completions.create.call_args
        self.assertGreater(call_kwargs[1]["temperature"], 0.0)


class TestModelFactory(unittest.TestCase):
    """Test MiniMax integration in ModelFactory."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_factory_creates_minimax_model(self):
        from camel.models import ModelFactory
        from camel.models.minimax_model import MiniMaxModel
        from camel.types import ModelPlatformType, ModelType
        model = ModelFactory.create(
            model_platform=ModelPlatformType.MINIMAX,
            model_type=ModelType.MINIMAX_M27,
            model_config_dict={"temperature": 0.2},
        )
        self.assertIsInstance(model, MiniMaxModel)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_factory_creates_minimax_highspeed(self):
        from camel.models import ModelFactory
        from camel.models.minimax_model import MiniMaxModel
        from camel.types import ModelPlatformType, ModelType
        model = ModelFactory.create(
            model_platform=ModelPlatformType.MINIMAX,
            model_type=ModelType.MINIMAX_M27_HIGHSPEED,
            model_config_dict={},
        )
        self.assertIsInstance(model, MiniMaxModel)

    def test_factory_rejects_wrong_platform_model_pair(self):
        from camel.models import ModelFactory
        from camel.types import ModelPlatformType, ModelType
        with self.assertRaises(ValueError):
            ModelFactory.create(
                model_platform=ModelPlatformType.MINIMAX,
                model_type=ModelType.GPT_4O,
                model_config_dict={},
            )


class TestNanoGraphRAGMiniMax(unittest.TestCase):
    """Test MiniMax functions in nano_graphrag._llm."""

    def test_minimax_complete_if_cache_exists(self):
        from nano_graphrag._llm import minimax_complete_if_cache
        self.assertTrue(callable(minimax_complete_if_cache))

    def test_minimax_m27_complete_exists(self):
        from nano_graphrag._llm import minimax_m27_complete
        self.assertTrue(callable(minimax_m27_complete))

    def test_minimax_m27_highspeed_complete_exists(self):
        from nano_graphrag._llm import minimax_m27_highspeed_complete
        self.assertTrue(callable(minimax_m27_highspeed_complete))

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_minimax_complete_if_cache_clamps_temperature(self):
        """Test that temperature is clamped in minimax_complete_if_cache."""
        from nano_graphrag._llm import minimax_complete_if_cache

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test"

        async def _run():
            with patch('nano_graphrag._llm.AsyncOpenAI') as MockClient:
                mock_instance = MagicMock()
                mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
                MockClient.return_value = mock_instance
                await minimax_complete_if_cache(
                    "MiniMax-M2.7", "test prompt", temperature=0.0
                )
                call_kwargs = mock_instance.chat.completions.create.call_args
                self.assertGreaterEqual(call_kwargs[1].get("temperature", 0.01), 0.01)

        asyncio.get_event_loop().run_until_complete(_run())


class TestNanoGraphRAGMiniMaxCaching(unittest.TestCase):
    """Test MiniMax caching in nano_graphrag._llm."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_cache_hit_returns_cached_value(self):
        """Test that cached results are returned without API call."""
        from nano_graphrag._llm import minimax_complete_if_cache

        mock_kv = MagicMock()
        mock_kv.get_by_id = AsyncMock(return_value={"return": "cached result"})

        async def _run():
            with patch('nano_graphrag._llm.AsyncOpenAI'):
                result = await minimax_complete_if_cache(
                    "MiniMax-M2.7", "test prompt", hashing_kv=mock_kv
                )
                self.assertEqual(result, "cached result")

        asyncio.get_event_loop().run_until_complete(_run())


if __name__ == "__main__":
    unittest.main()
