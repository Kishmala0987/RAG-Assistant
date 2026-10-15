"""
Command line demo of the HR Policy Agent
"""
from src.pipeline import ask, build_hr_agent
from src.logger import get_logger

logger = get_logger(__name__)
def main():
    """
    Building the HR policy agent
    """
    agent = build_hr_agent()
    demo_question = [
        "How many paid leaves do I get?",
        "What is the process of resignation?",
        "What is the promotion policy?",
        "Can I work home every day?",
        "What if someone needs sick leave even though they have already utilized all of his leaves"
    ]
    for question in demo_question:
        print("="*60)
        print("Question:", question)
        answer = ask(agent, question)
        print("="*60)
        print("AI Answer:", answer)
        print("="*60)
if __name__ == "__main__":
    main()