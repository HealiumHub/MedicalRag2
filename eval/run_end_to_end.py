import json
import pandas as pd
from llama_index.core import (
    Response,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.evaluation import (
    EvaluationResult,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from tqdm import tqdm
from const import PromptConfig
from eval.run import RerankedRetriever

from generations.completion import get_answer_with_context
from ingestion.dataloader import ingestion_index

LLM_MODEL = "gpt-3.5-turbo"
RETRIEVAL_TOP_K = 50
RERANK_TOP_K = 10

llm = OpenAI(temperature=0, model=LLM_MODEL)
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)
retriever = ingestion_index.read_from_chroma().as_retriever(
    similarity_top_k=RETRIEVAL_TOP_K
)
reranked_retriever = RerankedRetriever(retriever, top_k=RERANK_TOP_K)


# define jupyter display function
def display_eval_df(
    faithfulness_eval_results: list[EvaluationResult],
    relevancy_eval_results: list[EvaluationResult],
) -> pd.DataFrame:
    eval_df = pd.DataFrame(
        {
            "Question": [],
            "Response": [],
            "Source": [],
            "Faithfulness Evaluation Result": [],
            "Faithfulness Score": [],
            "Faithfulness Reasoning": [],
            "Relevancy Evaluation Result": [],
            "Relevancy Score": [],
            "Relevancy Reasoning": [],
        },
    )
    for faithfulness_eval_result, relevancy_eval_result in zip(
        faithfulness_eval_results, relevancy_eval_results
    ):
        eval_df = pd.concat(
            [
                eval_df,
                pd.DataFrame(
                    [
                        {
                            "Question": faithfulness_eval_result.query,
                            "Response": str(faithfulness_eval_result.response),
                            "Source": json.dumps(
                                faithfulness_eval_result.contexts,
                                indent=4,
                                ensure_ascii=False,
                            ),
                            "Faithfulness Evaluation Result": "Pass"
                            if faithfulness_eval_result.passing
                            else "Fail",
                            "Faithfulness Score": faithfulness_eval_result.score,
                            "Faithfulness Reasoning": faithfulness_eval_result.feedback,
                            "Relevancy Evaluation Result": "Pass"
                            if relevancy_eval_result.passing
                            else "Fail",
                            "Relevancy Score": relevancy_eval_result.score,
                            "Relevancy Reasoning": relevancy_eval_result.feedback,
                        }
                    ]
                ),
            ]
        )

    eval_df.to_csv(path_or_buf="end_to_end_eval_df.csv")
    return eval_df


# Read questions in pg_eval_dataset.json
questions = []
with open("pg_eval_dataset.json", "r") as f:
    dataset = json.load(f)
    questions = list(dataset["queries"].values())

# Construct evaluations.
responses = []
faithfulness_eval_results = []
relevancy_eval_results = []
for q in tqdm(questions):
    contexts = reranked_retriever.retrieve(q)
    contexts = [node.text for node in contexts]
    answer = get_answer_with_context(
        query=q,
        model_name=LLM_MODEL,
        related_articles=contexts,
        custom_instruction=PromptConfig.PERSONALITY,
        temperature=0,
        stream_handler=None,
    )
    eval_result = faithfulness_evaluator.evaluate(q, response=answer, contexts=contexts)
    faithfulness_eval_results.append(eval_result)

    eval_result = relevancy_evaluator.evaluate(q, response=answer, contexts=contexts)
    relevancy_eval_results.append(eval_result)

    responses.append(answer)

# Export to df.
eval_df = display_eval_df(faithfulness_eval_results, relevancy_eval_results)
print(eval_df.describe())
