# Study tokenizer

Frozen copy of the summer SLM byte-level BPE tokenizer, reused by both
architectures. Vocabulary: 50,000; EOS: 2; NFC normalization. No tokenizer
training is performed on the new study splits. The original tokenizer's
training corpus is not fully documented, so this is not a claim that its
vocabulary was fitted exclusively on the new training split.

SHA-256 of `tokenizer.json`:
`ac598311c740d555cb6f4f63c1a3cdd4d4cf63b7f61685481512ab6e4910517b`

The data preparer appends one EOS after each document and packs each split
separately. These loss values must not be equated with the summer runs'
validation losses, which used a different validation corpus.
