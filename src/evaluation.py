"""
Step 9: evaluate answer quality against a fixed set 
of test questions.

Unlike tracing (which just records what happened), 
evaluation runs the
agent against a known set of question/reference-answer 
pairs and scores
each answer using a second LLM as a judge.
Results are uploaded to
LangSmith as a Dataset + Experiment, 
so quality can be compared across
runs (after a prompt change, a new model, a new guardrail, etc).

The judge model is routed through Portkey too,
using the same slug as
the main app's LLM (gateway.py's PRIMARY_PROVIDER) 
but a different
underlying model (JUDGE_MODEL_NAME) - 
so it isn't grading its own
output verbatim, without needing a second slug set up.
"""

from langchain_openai import ChatOpenAI
from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, RAG_GROUNDEDNESS_PROMPT
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from src import config
from src.logger import get_logger
from src.gateway  import PRIMARY_PROVIDER
from src.pipeline import ask, build_hr_agent
from src.vector_store_qdrant import get_semantic_retriever, load_vector_store
from src.rag import search_hr_policy
from src.chunks import get_all_chunks_from_qdrant, get_bm25_retriever
from src.reranking import get_reranker

logger = get_logger(__name__)

#question paper name
DATASET_NAME = "hr-policy-qna"

TEST_CASES = [
    {
        "question": "What is the standard working schedule for full-time employees?",
        "answer": "Eight working hours per day, five days per week. Individual departments may set different schedules based on operational requirements.",
    },
    {
        "question": "Who should an employee notify if they can't attend work due to illness or emergency, and when?",
        "answer": "Their immediate supervisor, as soon as reasonably possible, and ideally before their scheduled start time.",
    },
    {
        "question": "Is remote work an entitlement for employees?",
        "answer": "No. Remote work is a work arrangement, not an entitlement, and may be modified or withdrawn based on business requirements, employee performance, security concerns, or operational needs.",
    },
    {
        "question": "Can an employee assume their leave request is approved once they submit it?",
        "answer": "No. Employees should not assume approval until they receive confirmation through the appropriate channel.",
    },
    {
        "question": "What should an employee do if they find a discrepancy in their salary or deductions?",
        "answer": "Report it to HR as soon as possible. HR will investigate and provide clarification where necessary.",
    },
    {
        "question": "What are the rules on entering company data into external AI tools?",
        "answer": "Employees must not enter confidential customer information, private employee information, passwords, proprietary source code, unreleased business information, or other restricted data into external AI systems unless the organization has explicitly approved the tool and use case.",
    },
    {
        "question": "When must AI-generated content be reviewed by an employee?",
        "answer": "Before it is used for important business decisions, customer communications, legal documents, financial decisions, or other high-impact activities.",
    },
    {
        "question": "Does attending training guarantee a promotion or salary increase?",
        "answer": "No. Participation in training does not automatically guarantee promotion or salary adjustment.",
    },
    {
        "question": "What should an employee do when they become aware of a potential conflict of interest, and does it automatically mean a policy violation?",
        "answer": "They should disclose it to their manager or HR as soon as they become aware of it. A potential conflict does not automatically mean a violation; the organization will review it and decide how to manage it.",
    },
    {
        "question": "What must employees do when their employment ends?",
        "answer": "Return company property, follow instructions on returning or deleting confidential information on personal or company-managed devices, and comply with continuing obligations on confidentiality, intellectual property, and other applicable agreements.",
    },
]

JUDGE_MODEL_NAME = "openai/gpt-oss-20b"

#making a judge llm
def _get_judge_llm()->ChatOpenAI:
    return ChatOpenAI(
        api_key="portkey",
        base_url = PORTKEY_GATEWAY_URL,
        model = JUDGE_MODEL_NAME,
        default_headers=createHeaders(api_key = config.PORTKEY_API_KEY, provider = "@hrpolicybackup")
    )

def _ensure_dataset(client:Client):
    """Create the LangSmith dataset if it doesnt exist yet"""
    if client.has_dataset(dataset_name  = DATASET_NAME):
        return client.read_dataset(dataset_name = DATASET_NAME)
    dataset = client.create_dataset(dataset_name = DATASET_NAME)
    for case in TEST_CASES:
        client.create_example(
            dataset_id = dataset.id,
            inputs= {"question": case["question"]},
            outputs = {"answer": case["answer"]}
        )
    
    return dataset


def run_evaluation():
    client = Client()
    dataset = _ensure_dataset(client)

    agent = build_hr_agent()

    semantic_retriever = get_semantic_retriever(load_vector_store()) 
    keyword_retriever = get_bm25_retriever(get_all_chunks_from_qdrant())
    reranker = get_reranker()

    def target(inputs:  dict) -> dict:
        answer = ask(agent, inputs["question"])
        context = search_hr_policy(inputs["question"], semantic_retriever, keyword_retriever, reranker)
        return {"answer": answer, "context": context}
    #giving marks
    correctness_evaluator = create_llm_as_judge(
        prompt = CORRECTNESS_PROMPT,
        feedback_key="correctness"  ,
        judge=_get_judge_llm()      
    )

    groundedness_judge = create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT, 
        feedback_key="groundedness",
        judge=_get_judge_llm()
    )

    def groundedness_evaluator(outputs:dict, **kwargs) -> dict :
        """Check the answer is checked by retrieved context"""
        return groundedness_judge(outputs={"answer": outputs["answer"]}, context=outputs["context"])
    return client.evaluate(
        target,
        data = dataset.name,
        evaluators=[correctness_evaluator, groundedness_evaluator],
        experiment_prefix="hr-policy-evalzz",
        description = "HR Policy evaluator : correctnesss+  groundedness",
        max_concurrency = 2
    )
