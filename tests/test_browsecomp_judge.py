from benchmarks.browsecomp_judge import JUDGE_PROMPT, parse_verdict


def test_judge_prompt_keeps_canonical_wording() -> None:
    rendered = JUDGE_PROMPT.format(question="Q", response="R", correct_answer="A")
    assert rendered.startswith(
        "Judge whether the following [response] to [question] is correct or not based on "
        "the precise and unambiguous [correct_answer] below."
    )
    assert "if there if there is any inconsistency" in rendered
    assert rendered.endswith(
        "confidence: The extracted confidence score between 0% and 100% from [response]. "
        "Put 100 if there is no confidence score available.\n"
    )


def test_parse_verdict_accepts_case_and_markdown_bold() -> None:
    assert parse_verdict("**correct:** **YES**\n**confidence:** **73%**") == (True, 73)
    assert parse_verdict("**correct**: no\nconfidence: 100") == (False, 100)


def test_parse_verdict_uses_last_correct_field() -> None:
    response = "reasoning: The draft said correct: no.\ncorrect: yes\nconfidence: 82%"
    assert parse_verdict(response) == (True, 82)


def test_parse_verdict_defaults_unparseable_correct_to_no() -> None:
    assert parse_verdict("The answer looks right.") == (False, 100)
