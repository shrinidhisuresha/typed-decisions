import pytest

WORDS = ["refund", "charge", "invoice", "crash", "error", "bug", "ticket", "the", "my"]


@pytest.fixture(scope="session")
def tiny_base(tmp_path_factory):
    """A 2-layer BERT built from a config: no network, trains in well under a second."""
    path = tmp_path_factory.mktemp("tiny-bert")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "{", "}", '"', ":", ","] + WORDS
    core = Tokenizer(models.WordLevel({w: i for i, w in enumerate(vocab)}, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=core, pad_token="[PAD]", unk_token="[UNK]")
    tokenizer.save_pretrained(path)
    config = transformers.BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                                     num_attention_heads=2, intermediate_size=64)
    transformers.BertForSequenceClassification(config).save_pretrained(path)
    return str(path)


