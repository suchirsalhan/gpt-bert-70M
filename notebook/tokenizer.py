

import argparse
import json
from collections import Counter
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer, pre_tokenizers, decoders, normalizers, Regex, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from huggingface_hub import login


def initialize_tokenizer(args):
    start_of_text_symbol = "<s>"
    end_of_text_symbol = "</s>"
    unk_symbol = "<unk>"
    mask_symbol = "<mask>"
    pad_symbol = "<pad>"

    special_tokens = [unk_symbol, start_of_text_symbol, end_of_text_symbol, pad_symbol, mask_symbol]
    special_tokens += [f"<special_{i}>" for i in range(11)]

    tokenizer = Tokenizer(BPE(
        unk_token=unk_symbol,
        byte_fallback=False,
        fuse_unk=False,
        ignore_merges=True
    ))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Prepend(" "),
        normalizers.NFKC(),
        normalizers.Replace(Regex("\n"), '\n '),
        normalizers.Replace(Regex(" *\n"), '\n'),
    ])

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            Regex("[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*| ?\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"),
            behavior="isolated",
            invert=False
        ),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False, trim_offsets=True),
        pre_tokenizers.Split(Regex(".{1,24}"), behavior="isolated", invert=False)
    ])

    tokenizer.decoder = decoders.Sequence([
        decoders.ByteLevel(add_prefix_space=False, use_regex=False),
        decoders.Strip(' ', 1, 0),
        decoders.Replace("\n ", "\n")
    ])

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{start_of_text_symbol} $A",
        pair=f"{start_of_text_symbol} $A {start_of_text_symbol} $B",
        special_tokens=[(start_of_text_symbol, 1)],
    )

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
        min_frequency=args.min_frequency
    )

    return tokenizer, trainer


def calculate_stats(tokenizer, dataset, args):
    counter, n_words = Counter(), 0
    for i, example in enumerate(tqdm(dataset)):
        text = example["text"].strip()
        if text:
            n_words += len(text.split())
            encoding = tokenizer.encode(text)
            tokens = encoding.tokens
            counter.update(tokens)

            if i == 0:
                print("Example of tokenization:")
                print(text)
                print(tokenizer.decode(encoding.ids))
                for j in encoding.ids:
                    print(j, tokenizer.id_to_token(j))

    sorted_subwords = counter.most_common()
    n_subwords = sum(freq for _, freq in sorted_subwords)
    print(f"Average splits per word: {n_subwords / n_words:.3f}")
    f_95 = sorted_subwords[len(sorted_subwords) * 95 // 100][1]
    print(f"F_{{95%}} is {f_95}\n")

    with open(f"{args.vocab_path.with_suffix('')}_stats.txt", "w") as f:
        f.write(f"Vocabulary size: {args.vocab_size}\n")
        f.write(f"Average splits per word: {n_subwords / n_words:.3f}\n")
        f.write(f"F_{{95%}} is {f_95}\n")
        sorted_str = '\n\t'.join(f"{freq}: {subword}" for subword, freq in sorted_subwords)
        f.write(f"Sorted subwords:\n\t{sorted_str}\n")


def test_examples(tokenizer):
    def test(text):
        return ' '.join(tokenizer.encode(text).tokens)

    texts = [
        "Robert built a 100 lbf liquid engine in 2001.",
        "what are examples of query interfaces like SQL or XPath?",
        "I'm a sociophonetician who works on prosody!",
        "The Northern Lights season is here...",
        "Some people have SOTA facial recognition abilities.",
    ]

    for text in texts:
        print(f"INPUT:  {text}\nTOKENS: {test(text)}\n")


def save_and_upload_to_hub(tokenizer, args):
    # Save raw tokenizer JSON
    tokenizer.save(str(args.vocab_path))

    # Clean added_tokens list
    with args.vocab_path.open("r") as f:
        tokenizer_json = json.load(f)
    tokenizer_json["added_tokens"] = tokenizer_json["added_tokens"][:-256]
    with args.vocab_path.open("w") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=4)

    print("Reloading and wrapping in PreTrainedTokenizerFast...")
    tokenizer = Tokenizer.from_file(str(args.vocab_path))

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        cls_token="<s>",
        sep_token="</s>",
    )

    print("Pushing to Hugging Face Hub...")
    fast_tokenizer.push_to_hub("gptbert-babylm-tokenizer", private=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', type=Path, default=Path("tokenizer_babylm70M.json"))
    parser.add_argument('--vocab_size', type=int, default=16384)
    parser.add_argument('--min_frequency', type=int, default=10)
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--hf_token', type=str, default=None, help="Hugging Face token (or set via HF_TOKEN env var)")

    # Accept unknown args to prevent Colab crashes
    args, unknown = parser.parse_known_args()

    # Hugging Face login
    if args.hf_token:
        login(token=args.hf_token)

    print("Loading dataset...", flush=True)
    dataset = load_dataset("Talking-Babies/babylm70M_text", split=args.split)

    print("Initializing tokenizer...", flush=True)
    tokenizer, trainer = initialize_tokenizer(args)

    print("Training tokenizer...", flush=True)
    tokenizer.train_from_iterator((x["text"] for x in tqdm(dataset)), trainer=trainer)

    print("Saving tokenizer and uploading...", flush=True)
    save_and_upload_to_hub(tokenizer, args)

    print("Calculating stats...", flush=True)
    tokenizer = Tokenizer.from_file(str(args.vocab_path))
    calculate_stats(tokenizer, dataset, args)

    print("Showing test examples...")
    test_examples(tokenizer)
