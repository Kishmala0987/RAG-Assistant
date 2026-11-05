"""
Run the correctness evalaution and upload the result to the langsmith
Run with: python evaluate.py
"""

#Display the scores

from src.evaluation import run_evaluation

def main():
    print("RUNNING THE HR POLICY ASSISTANT EVALUATION....")
    results = run_evaluation()
    print("OPEN LANGSMITH AND CHECK THE EXPERIMENT....")
    print(results)
if __name__ == "__main__":
    main()