import sys
from types import SimpleNamespace

import pytest

from src.llm import get_llm


@pytest.mark.parametrize('configured,expected', [
    ('llama-3.1-8b-instant', 'openai/gpt-oss-20b'),
    ('llama-3.3-70b-versatile', 'openai/gpt-oss-120b'),
    ('qwen/qwen3.6-27b', 'qwen/qwen3.6-27b'),
])
def test_hosted_model_overrides_migrate_retired_ids(monkeypatch, configured, expected):
    calls = []
    monkeypatch.setenv('GROQ_API_KEY', 'test-placeholder')
    monkeypatch.setenv('GROQ_MODEL', configured)
    monkeypatch.setitem(sys.modules, 'langchain_groq', SimpleNamespace(ChatGroq=lambda **kwargs: calls.append(kwargs)))
    llm = get_llm()
    assert calls[0]['model'] == expected
    assert llm.provider == f'groq:{expected}'
