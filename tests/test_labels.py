import pytest
from typed_decisions.labels import allocate_labels, LABEL_ALPHABET, LabelError


class FakeTokenizer:
    """Single token per ' X' form, which is what a well-behaved BPE tokenizer does."""

    def encode(self, text: str) -> list[int]:
        if text.startswith(" ") and len(text) == 2 and text[1] in LABEL_ALPHABET:
            return [1000 + LABEL_ALPHABET.index(text[1])]
        return [7] * len(text)


class SplittingTokenizer(FakeTokenizer):
    """Pathological: splits ' C' into two tokens."""

    def encode(self, text: str) -> list[int]:
        if text == " C":
            return [55, 66]
        return super().encode(text)


def test_allocates_one_label_per_option_in_order():
    labels = allocate_labels(["red", "green", "blue"], FakeTokenizer())
    assert [l.label for l in labels] == ["A", "B", "C"]


def test_labels_carry_a_leading_space_so_the_prompt_can_end_without_one():
    labels = allocate_labels(["red"], FakeTokenizer())
    assert labels[0].rendered == " A"


def test_each_label_maps_to_a_distinct_token_id():
    labels = allocate_labels(["a", "b", "c", "d"], FakeTokenizer())
    ids = [l.token_id for l in labels]
    assert len(set(ids)) == len(ids)


def test_labels_are_paired_with_their_option():
    labels = allocate_labels(["red", "green"], FakeTokenizer())
    assert [l.option for l in labels] == ["red", "green"]


def test_rejects_more_options_than_the_alphabet_can_label():
    too_many = [f"opt{i}" for i in range(len(LABEL_ALPHABET) + 1)]
    with pytest.raises(LabelError, match="retrieval"):
        allocate_labels(too_many, FakeTokenizer())


def test_rejects_a_label_the_tokenizer_splits_into_multiple_tokens():
    with pytest.raises(LabelError, match="single token"):
        allocate_labels(["a", "b", "c"], SplittingTokenizer())


def test_rejects_an_empty_option_list():
    with pytest.raises(LabelError, match="at least"):
        allocate_labels([], FakeTokenizer())


def test_rejects_duplicate_options():
    with pytest.raises(LabelError, match="duplicate"):
        allocate_labels(["red", "red"], FakeTokenizer())
