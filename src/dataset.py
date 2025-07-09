import torch
import random

"""
- Span-based masking with probabilistic replacement. Dynamic multi-segment packing for training efficiency.
Token usage tracking to reduce overfitting to frequent spans.
Causal and masked dataset modes to suit different model objectives.
"""

"""

🔹 SpanMaskingStrategy
This class defines how to mask tokens in spans, inspired by span-based masking from models like T5 and SpanBERT.

Parameters:
n_special_tokens: ID threshold to avoid masking special tokens.

random_p, keep_p: Probabilities for how to replace masked tokens.

With random_p: Replace with random token.

With keep_p: Keep original token.

Otherwise: Use the <mask> token.

vocab_size: For sampling random tokens.

mask_token_id: ID for the <mask> token.

max_span_length: Maximum length of spans to mask (default: 3).

Key Steps in __call__():
Generate spans: Sample span lengths randomly.

Map tokens to spans via cumulative indexing.

Compute masking scores (mask_ratios) based on random noise and token frequency (counts).

Decide how to replace tokens:

Random token if replacement_p < random_p

Keep original if random_p <= replacement_p <= random_p + keep_p

Use <mask> otherwise.

Returns:

mask_ratios: Scores used to select which tokens to mask.

replacement_tokens: The actual input tokens after masking.

🔹 RandomIndex
Randomizes access to segments for dynamic packing.

Ensures all segments get sampled evenly before repetition (via reshuffling).
"""


class SpanMaskingStrategy:
    def __init__(self, n_special_tokens, random_p, keep_p, vocab_size, mask_token_id):
        self.n_special_tokens = n_special_tokens
        self.random_p = random_p
        self.keep_p = keep_p
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.max_span_length = 3

    def __call__(self, tokens, counts=None):
        length = tokens.size(0)

        span_lengths = torch.randint(1, self.max_span_length + 1, size=(length,), dtype=torch.int)
        cumsum = torch.cumsum(span_lengths, dim=0)

        total_length = cumsum[-1].item()
        indices = torch.zeros(total_length, dtype=torch.int)
        indices[cumsum - span_lengths] = torch.arange(length, dtype=torch.int)
        indices = torch.cummax(indices, dim=0)[0]
        indices = indices[:length]

        max_index = indices[-1].item()
        span_random_numbers_1, span_random_numbers_2 = torch.rand([(max_index + 1) * 2]).chunk(2)

        mask_ratios = span_random_numbers_1[indices]

        if counts is not None:
            counts = counts.float()
            counts[tokens < self.n_special_tokens] = float('-inf')
            counts_p = torch.nn.functional.softmax(counts, dim=0)
            mask_ratios = mask_ratios * counts_p

        mask_ratios[tokens < self.n_special_tokens] = float('inf')

        replacement_p = span_random_numbers_2[indices]
        random_mask = replacement_p < self.random_p

        replacement_tokens = tokens.clone()
        replacement_tokens[random_mask] = torch.randint(
            low=self.n_special_tokens,
            high=self.vocab_size,
            size=[random_mask.sum().item()],
            dtype=torch.long
        )
        replacement_tokens[replacement_p > (self.random_p + self.keep_p)] = self.mask_token_id

        return mask_ratios, replacement_tokens


class RandomIndex:
    def __init__(self, n_segments):
        self.n_segments = n_segments
        self.indices = torch.randperm(n_segments)
        self.index = 0

    def get_random_index(self):
        if self.index >= self.n_segments:
            self.indices = torch.randperm(self.n_segments)
            self.index = 0

        index = self.indices[self.index]
        self.index += 1

        return index


"""
Used for masked language modeling (MLM).

Key Features:
Loads tokenized documents from disk (torch.load(input_file)).

Splits documents into fixed-length segments.

Tracks how often each token is seen/masked (self.counts and self.mask_counts).

Dynamically packs multiple segments into one sequence, separated by <s> (CLS token).

Applies masking via SpanMaskingStrategy.

__getitem__() Workflow:
Select a segment (index).

Apply span masking with dynamic mask_p depending on global_step.

Pack additional segments if sequence is too short, sampling from other segments using RandomIndex.

Preference given to least-seen spans (argmin(summed_counts)).

Pad to seq_length.

Create attention mask:

Full attention (block diagonal) across segments.

Inverted (~attention_mask) to suit transformer conventions.

apply_mask():
Applies a dynamic masking probability (mask_p) computed from training progress (global_step).

Selects the lowest mask_ratios up to a percentile threshold.

Outputs masked inputs, targets (labels), and actual masked ratio.
"""

class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, input_file: str, tokenizer, args, seq_length, rank, world_size):
        self.path = input_file
        self.seq_length = seq_length
        self.n_special_tokens = args.n_special_tokens
        self.args = args
        self.global_step = 0

        self.mask_index = tokenizer.token_to_id("<mask>")
        self.cls_index = tokenizer.token_to_id("<s>")
        self.pad_index = tokenizer.token_to_id("<pad>")

        self.masking_strategy = SpanMaskingStrategy(args.n_special_tokens, args.mask_random_p, args.mask_keep_p, args.vocab_size, self.mask_index)

        documents = torch.load(input_file)
        self.segments = [
            document[offset : offset + self.seq_length - 2]
            for document in documents
            for offset in range(0, len(document), self.seq_length - 2)
            if len(document) > 0 and len(document) - offset > 1
        ]
        if rank is not None:
            self.segments = self.segments[rank::world_size]
        self.counts = [
            torch.zeros_like(segment)
            for segment in self.segments
        ]
        self.mask_counts = [
            torch.zeros_like(segment)
            for segment in self.segments
        ]
        self.random_index = RandomIndex(len(self.segments))

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        tokens = self.segments[index]
        seq_length = min(self.seq_length, tokens.size(0))
        tokens = tokens[:seq_length].long()
        self.counts[index][:seq_length] += 1

        mask_ratios, replacement_tokens = self.masking_strategy(tokens, self.mask_counts[index][:seq_length])
        input_ids, target_ids, real_mask_p = self.apply_mask(tokens, mask_ratios, replacement_tokens)
        self.mask_counts[index][:seq_length][target_ids != -100] += 1

        input_ids = torch.cat([
            torch.LongTensor([self.cls_index]),
            input_ids
        ])
        target_ids = torch.cat([
            torch.LongTensor([-100]),
            target_ids
        ])
        attention_mask = torch.ones(seq_length + 1, seq_length + 1, dtype=torch.bool)

        while self.seq_length - input_ids.size(0) > 1:
            index = self.random_index.get_random_index()
            tokens = self.segments[index].long()
            seq_length = min(self.seq_length - input_ids.size(0), tokens.size(0))

            # select random offset
            offset = 0
            if seq_length < tokens.size(0):
                conv_weight = torch.ones(1, 1, seq_length)
                summed_counts = torch.nn.functional.conv1d(
                    self.counts[index].view(1, 1, -1).float(),
                    conv_weight
                ).squeeze()
                offset = torch.argmin(summed_counts)

            tokens = tokens[offset:offset + seq_length]
            self.counts[index][offset:offset+seq_length] += 1

            mask_ratios, replacement_tokens = self.masking_strategy(tokens, self.mask_counts[index][offset:offset+seq_length])
            input_ids_, target_ids_, _ = self.apply_mask(tokens, mask_ratios, replacement_tokens)

            input_ids = torch.cat([
                input_ids,
                torch.LongTensor([self.cls_index]),
                input_ids_,
            ])
            target_ids = torch.cat([
                target_ids,
                torch.LongTensor([-100]),
                target_ids_
            ])
            attention_mask = torch.block_diag(
                attention_mask,
                torch.ones(seq_length + 1, seq_length + 1, dtype=torch.bool)
            )

            self.mask_counts[index][offset:offset+seq_length][target_ids_ != -100] += 1

        padding_length = self.seq_length - input_ids.size(0) + 1
        if padding_length > 0:
            input_ids = torch.cat([
                input_ids,
                torch.LongTensor([self.pad_index] * padding_length)
            ])
            target_ids = torch.cat([
                target_ids,
                torch.LongTensor([-100] * padding_length)
            ])
            attention_mask = torch.block_diag(
                attention_mask,
                torch.zeros(padding_length, padding_length, dtype=torch.bool)
            )

        attention_mask = ~attention_mask

        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        attention_mask = attention_mask[:-1, :-1]

        return input_ids, target_ids, attention_mask, real_mask_p

    def set_global_step(self, global_step):
        self.global_step = global_step

    def apply_mask(self, input_ids, mask_ratios, replacement_ids):
        mask_p = self.args.mask_p_start + (self.args.mask_p_end - self.args.mask_p_start) * self.global_step / self.args.max_steps
        mask_p = torch.topk(mask_ratios, max(1, int(mask_ratios.size(0) * mask_p + torch.rand(1).item())), largest=False).values.max().item()

        mask = mask_ratios <= mask_p
        target_ids = torch.where(mask, input_ids, -100)
        input_ids = torch.where(mask, replacement_ids, input_ids)

        real_mask_p = mask.sum() / mask_ratios.numel()

        return input_ids, target_ids, real_mask_p

    def show_random_item(self, tokenizer):
        input_ids, target_ids, attention_mask, real_mask_p = self.__getitem__(torch.randint(0, len(self), []).item())
        print(' '.join(tokenizer.id_to_token(i) for i in input_ids.tolist()), flush=True)
        print()
        print(' '.join(str(i) for i in input_ids.tolist()), flush=True)
        print()
        print(' '.join(tokenizer.id_to_token(i) if i != -100 else "-100" for i in target_ids.tolist()), flush=True)
        print()
        print(real_mask_p, flush=True)


"""
🔹 CausalDataset
Used for causal language modeling (CLM), like GPT.

Key Differences from MaskedDataset:
No token masking or span replacement.

Inputs and targets are simply the next-token prediction targets.

Attention mask is made causal (lower triangular) with attention_mask.tril().

Purpose:
Trains a model to predict the next token given prior context, using dynamic packing and <s> separators.

⚙️ Dynamic Packing (in both datasets)
Instead of returning just one segment, each item:

Packs multiple short segments together until the desired seq_length is met.

Uses <s> tokens as boundaries between segments.

Updates attention masks to prevent cross-segment attention (in MLM) or enforce causality (in CLM).

🧪 show_random_item()
Diagnostic method to print a random example from MaskedDataset with:

Input tokens (after masking)

Corresponding labels (with -100 where loss should be ignored)

Actual mask ratio
"""

class CausalDataset(torch.utils.data.Dataset):
    def __init__(self, input_file: str, tokenizer, args, seq_length, rank, world_size):
        self.path = input_file
        self.seq_length = seq_length
        self.n_special_tokens = args.n_special_tokens
        self.args = args
        self.global_step = 0

        self.mask_index = tokenizer.token_to_id("<mask>")
        self.cls_index = tokenizer.token_to_id("<s>")
        self.pad_index = tokenizer.token_to_id("<pad>")

        documents = torch.load(input_file)
        self.segments = [
            document[offset : offset + self.seq_length - 2]
            for document in documents
            for offset in range(0, len(document), self.seq_length - 2)
            if len(document) > 0 and len(document) - offset > 1
        ]
        if rank is not None:
            self.segments = self.segments[rank::world_size]
        self.counts = [
            torch.zeros_like(segment)
            for segment in self.segments
        ]
        self.random_index = RandomIndex(len(self.segments))

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        tokens = self.segments[index]
        seq_length = min(self.seq_length, tokens.size(0))
        self.counts[index][:seq_length] += 1

        input_ids = torch.cat([
            torch.LongTensor([self.cls_index]),
            tokens[:seq_length].long()
        ])
        target_ids = torch.cat([
            torch.LongTensor([-100]),
            tokens[:seq_length].long()
        ])
        attention_mask = torch.ones(seq_length + 1, seq_length + 1, dtype=torch.bool)

        while self.seq_length - input_ids.size(0) > 1:
            index = self.random_index.get_random_index()
            tokens = self.segments[index].long()
            seq_length = min(self.seq_length - input_ids.size(0), tokens.size(0))

            # select random offset
            offset = 0
            if seq_length < tokens.size(0):
                conv_weight = torch.ones(1, 1, seq_length)
                summed_counts = torch.nn.functional.conv1d(
                    self.counts[index].view(1, 1, -1).float(),
                    conv_weight
                ).squeeze()
                offset = torch.argmin(summed_counts)

            tokens = tokens[offset:offset + seq_length]
            self.counts[index][offset:offset+seq_length] += 1

            input_ids = torch.cat([
                input_ids,
                torch.LongTensor([self.cls_index]),
                tokens
            ])
            target_ids = torch.cat([
                target_ids,
                torch.LongTensor([-100]),
                tokens
            ])
            attention_mask = torch.block_diag(
                attention_mask,
                torch.ones(seq_length + 1, seq_length + 1, dtype=torch.bool)
            )

        padding_length = self.seq_length - input_ids.size(0) + 1
        if padding_length > 0:
            input_ids = torch.cat([
                input_ids,
                torch.LongTensor([self.pad_index] * padding_length)
            ])
            target_ids = torch.cat([
                target_ids,
                torch.LongTensor([-100] * padding_length)
            ])
            attention_mask = torch.block_diag(
                attention_mask,
                torch.zeros(padding_length, padding_length, dtype=torch.bool)
            )

        # make the attention mask causal
        attention_mask = attention_mask.tril()
        attention_mask = ~attention_mask

        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        attention_mask = attention_mask[:-1, :-1]

        return input_ids, target_ids, attention_mask, torch.zeros([])

    def set_global_step(self, global_step):
        self.global_step = global_step

    def show_random_item(self, tokenizer):
        input_ids, target_ids, attention_mask, real_mask_p = self.__getitem__(torch.randint(0, len(self), []).item())
        print(' '.join(tokenizer.id_to_token(i) for i in input_ids.tolist()), flush=True)
        print()
        print(' '.join(str(i) for i in input_ids.tolist()), flush=True)
        print()
        print(' '.join(tokenizer.id_to_token(i) if i != -100 else "-100" for i in target_ids.tolist()), flush=True)
        print()
        print(real_mask_p, flush=True)


class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, input_file: str, tokenizer, args):
        self.path = input_file
        self.seq_length = args.seq_length
        self.n_special_tokens = args.n_special_tokens

        self.mask_index = tokenizer.token_to_id("<mask>")
        self.cls_index = tokenizer.token_to_id("<s>")
        self.sep_index = tokenizer.token_to_id("</s>")
        self.pad_index = tokenizer.token_to_id("<pad>")

        self.masking_strategy = SpanMaskingStrategy(args.n_special_tokens, args.mask_random_p, args.mask_keep_p, args.vocab_size, self.mask_index)

        documents = torch.load(input_file)
        self.segments = [
            document[offset : offset + self.seq_length - 2]
            for document in documents
            for offset in range(0, len(document), self.seq_length - 2)
            if len(document) > 0 and len(document) - offset > 1
        ]
        if hasattr(args, "rank"):
            self.segments = self.segments[args.rank::args.world_size]
            random.seed(args.rank)
        else:
            random.seed(args.seed)
        random.shuffle(self.segments)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        tokens = self.segments[index]
        seq_length = min(self.seq_length - 2, tokens.size(0))

        segment = torch.cat([
            torch.LongTensor([self.cls_index]),
            tokens[:seq_length].long()
        ])
        attention_mask = torch.ones(seq_length + 1, seq_length + 1, dtype=torch.bool)

        mask_ratios, replacement_tokens = self.masking_strategy(segment)
        input_ids, target_ids, real_mask_p = self.apply_mask(segment, mask_ratios, replacement_tokens)

        padding_length = self.seq_length - segment.size(0) + 1
        if padding_length > 0:
            input_ids = torch.cat([
                input_ids,
                torch.LongTensor([self.pad_index] * padding_length)
            ])
            target_ids = torch.cat([
                target_ids,
                torch.LongTensor([-100] * padding_length)
            ])
            attention_mask = torch.block_diag(
                attention_mask,
                torch.zeros(padding_length, padding_length, dtype=torch.bool)
            )

        attention_mask = ~attention_mask

        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        attention_mask = attention_mask[:-1, :-1]

        return input_ids, target_ids, attention_mask, real_mask_p

    def apply_mask(self, input_ids, mask_ratios, replacement_ids):
        mask_p = 0.15
        mask_p = torch.topk(mask_ratios, max(1, int(mask_ratios.size(0) * mask_p + 0.5)), largest=False).values.max().item()

        mask = mask_ratios < mask_p
        target_ids = torch.where(mask, input_ids, -100)
        input_ids = torch.where(mask, replacement_ids, input_ids)

        real_mask_p = mask.sum() / mask_ratios.numel()

        return input_ids, target_ids, real_mask_p
