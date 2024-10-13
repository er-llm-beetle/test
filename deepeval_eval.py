import pytest
from deepeval import assert_test, evaluate
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase

from dotenv import load_dotenv
load_dotenv()




#  --------------- DEEPEVAL EXAMPLE CODE w USAGE --------------------



import pytest
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric, ToxicityMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

def evaluate_llm_output(actual_answer, predicted_answer, question):
    # Define metrics
    correctness_metric = GEval(
        name='Correctness',
        criteria="Determine whether the actual output is factually correct based on the expected output.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
    bias_metric = BiasMetric(threshold=0.7)
    toxicity_metric = ToxicityMetric(threshold=0.7)

    # Create test case
    test_case = LLMTestCase(
        input=question,
        actual_output=predicted_answer,
        expected_output=actual_answer,
    )

    # Evaluate the test case
    scores = evaluate(
        [test_case], 
        [
            answer_relevancy_metric,
            bias_metric,
            toxicity_metric,
        ]
    )

    # Measure correctness
    correctness_metric.measure(test_case)

    # Prepare results
    results = {
        "answer_relevancy_score": answer_relevancy_metric.score,
        "bias_score": bias_metric.score,
        "toxicity_score": toxicity_metric.score,
        "correctness_score": correctness_metric.score,
    }

    return results



# Example usage
actual_answer = "İşsizlik sığortası, işsiz qalan şəxslərə müəyyən müddətə maddi yardım göstərən sosial sığorta növüdür. Bu sığorta işsiz şəxslərə işsiz qaldıqları müddətdə maliyyə dəstəyi verir."
predicted_answer = "İşsizlik sığortası işçiyə müxtəlif hallar da işsizlik şəraitində aid göndərilən maddi yardımı növüdür. İşsizlüyün vaxtı zamanında, işçi tərəfindən ödənilməli olduğu yaranacaq gəlirini əldə etmək üçün sığorta müdafiəsinə görə ödənilir."
question = "İşsizlik sığortası nədir və hansı hallarda ödənilir?"

scores = evaluate_llm_output(actual_answer, predicted_answer, question)
print(scores)


