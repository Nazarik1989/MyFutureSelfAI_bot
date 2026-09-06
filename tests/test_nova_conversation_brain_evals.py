import json
from pathlib import Path

from future_self.nova_brain import validate_memory_candidate
from future_self.nova_companion_flow import NovaCompanionDiscourseReducer
from future_self.nova_companion_handlers import _has_untrusted_operational_claim
from future_self.schemas import NovaCompanionMemoryCandidate


def test_nova_conversation_brain_eval_corpus_is_complete_deterministic_and_offline():
    path = Path(__file__).parent / "evals/nova_conversation_brain_cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(cases, list)
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(
        case.get("expected_route")
        in {
            "identity",
            "companion",
            "memory_recall",
            "memory_forget",
            "reminder",
        }
        for case in cases
    )
    assert {case["category"] for case in cases} == {
        "identity_recall",
        "profile_grounding",
        "short_continuation",
        "assistant_offer_continuation",
        "open_loop",
        "long_term_recall",
        "correction",
        "forget",
        "reminder_handoff",
        "no_false_execution_claim",
        "no_router_collision",
        "privacy_isolation",
    }

    for case in cases:
        prior_assistant = case.get("prior_assistant")
        if isinstance(prior_assistant, str):
            anchor = NovaCompanionDiscourseReducer.reduce(
                case["input"],
                [{"role": "assistant", "content": prior_assistant}],
            )
            expected_kind = case["expected_offer_kind"]
            assert (anchor.offer_kinds[0] if anchor is not None else None) == expected_kind
        memory = case.get("memory")
        if isinstance(memory, dict):
            candidate = NovaCompanionMemoryCandidate.model_validate(memory)
            validated = validate_memory_candidate(candidate, user_text=case["input"])
            assert (validated is not None) == (case["expected_memory"] == "accepted")
        if case.get("expected_claim_rejected"):
            assert _has_untrusted_operational_claim(
                case["input"],
                user_text="Напомни мне о стрижке",
                action_context=True,
            )
