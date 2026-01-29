"""Tests for LLM client."""

from unittest.mock import MagicMock, patch

import pytest

from agents.utils.llm_client import (
    GrokProvider,
    LLMClient,
    LLMResponse,
    OpenAIProvider,
    YandexProvider,
)


class TestLLMResponse:
    """Tests for LLMResponse model."""

    def test_llm_response(self):
        """Test LLMResponse model."""
        response = LLMResponse(
            content="Generated code",
            model="gpt-4o-mini",
            provider="openai",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        assert response.content == "Generated code"
        assert response.total_tokens == 150


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    def test_init_requires_api_key(self):
        """Test that initialization requires an API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                OpenAIProvider()

    @patch("agents.utils.llm_client.OpenAI")
    def test_generate(self, mock_openai_class):
        """Test generating content."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated content"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30

        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()
            result = provider.generate("Test prompt")

        assert result.content == "Generated content"
        assert result.provider == "openai"
        assert result.total_tokens == 30


class TestGrokProvider:
    """Tests for Grok provider."""

    def test_init_requires_api_key(self):
        """Test that initialization requires an API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Grok API key is required"):
                GrokProvider()

    @patch("agents.utils.llm_client.httpx.Client")
    def test_generate(self, mock_client_class):
        """Test generating content."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Grok response"}}],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15,
            },
        }
        mock_client.post.return_value = mock_response

        with patch.dict("os.environ", {"GROK_API_KEY": "test-key"}):
            provider = GrokProvider()
            result = provider.generate("Test prompt")

        assert result.content == "Grok response"
        assert result.provider == "grok"


class TestYandexProvider:
    """Tests for Yandex provider."""

    def test_init_requires_api_key(self):
        """Test that initialization requires an API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Yandex API key is required"):
                YandexProvider()

    def test_init_requires_folder_id(self):
        """Test that initialization requires a folder ID."""
        with patch.dict("os.environ", {"YANDEX_API_KEY": "key"}, clear=True):
            with pytest.raises(ValueError, match="Yandex folder ID is required"):
                YandexProvider()


class TestLLMClient:
    """Tests for LLMClient factory class."""

    def test_unknown_provider(self):
        """Test that unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMClient(provider="unknown")

    @patch("agents.utils.llm_client.OpenAI")
    def test_default_provider(self, mock_openai):
        """Test default provider is OpenAI."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = LLMClient()
            assert client.provider.provider_name == "openai"

    @patch("agents.utils.llm_client.OpenAI")
    def test_generate_with_retry(self, mock_openai_class):
        """Test retry logic on failure."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Success"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30

        mock_client.chat.completions.create.side_effect = [
            Exception("First fail"),
            mock_response,
        ]

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = LLMClient(max_retries=3, retry_delay=0.01)
            result = client.generate("Test prompt")

        assert result.content == "Success"
        assert mock_client.chat.completions.create.call_count == 2

    @patch("agents.utils.llm_client.OpenAI")
    def test_token_tracking(self, mock_openai_class):
        """Test token usage tracking."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30

        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = LLMClient()
            client.generate("Prompt 1")
            client.generate("Prompt 2")

        assert client.total_tokens_used == 60

    @patch("agents.utils.llm_client.OpenAI")
    def test_generate_code(self, mock_openai_class):
        """Test generate_code helper method."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "def hello(): pass"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30

        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = LLMClient()
            result = client.generate_code("Write a function", context="# context")

        assert result == "def hello(): pass"
