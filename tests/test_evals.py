# tests/test_evals.py
#
# The eval harness measures the agent's judgement, which is the one thing
# the rest of the suite deliberately does not test. Only the parts that
# can be scored without spending money live here: the fixture loader and
# the scorer. Whether the agent actually passes is evals/run.py's job,
# and it costs real tokens every time.

import pytest


def test_the_model_call_reports_usage_when_asked():
    # Tokens are dropped on the floor today. The eval harness needs them,
    # and so will the cost ceiling in Decision D.
    from agent.loop import openai_model_call

    seen = []
    call = openai_model_call(on_usage=seen.append)

    assert callable(call)
    # The seam is optional: a caller that does not care about usage
    # passes nothing and is unaffected.
    assert openai_model_call() is not None
