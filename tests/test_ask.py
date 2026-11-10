"""Smoke test:"""

from src.pipeline import build_hr_agent, ask

def test_ask_returns_a_Real_answer():
    agent = build_hr_agent()
    answer = ask(agent, "How many paid leaves do I get per year?")
    assert isinstance(answer, str)
    assert answer.strip() != ""