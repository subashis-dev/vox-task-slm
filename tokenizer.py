"""
Comprehensive Subword & Character Tokenizer for Top-Tier Voice Assistant SLM.
Includes tokens for numbers (digits & spelled out), AM/PM variants, time idioms,
and assistant action keys.
"""

import json
from typing import List


class VoiceTaskTokenizer:
    SPECIAL_TOKENS = {
        "<pad>": 0,
        "<bos>": 1,
        "<sep>": 2,
        "<eos>": 3,
    }

    # High-frequency keywords and time words for single-token efficiency
    KEYWORDS = [
        # JSON Schema keys
        '{"action":',
        '"REMIND"',
        '"TODO"',
        '"DONE"',
        '"DELETE"',
        '"LIST"',
        '"GREET"',
        ',"task":',
        ',"time":',
        ',"category":',
        ',"filter":',
        ',"reply":',

        # Days & Relative periods
        'tomorrow',
        'today',
        'tonight',
        'yesterday',
        'morning',
        'afternoon',
        'evening',
        'night',
        'noon',
        'midnight',
        'monday',
        'tuesday',
        'wednesday',
        'thursday',
        'friday',
        'saturday',
        'sunday',

        # Spoken numbers & time markers
        'o\'clock',
        'in the morning',
        'in the afternoon',
        'in the evening',
        'at night',
        'am',
        'pm',
        'a.m.',
        'p.m.',
        ':00',
        ':15',
        ':30',
        ':45',
        ':20',
        ':10',
        ':50',

        # Spelled-out number words
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
        'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'twenty',
        'thirty', 'forty', 'fifty', 'half', 'quarter',

        # Units
        'minutes',
        'minute',
        'hours',
        'hour',
        'days',
        'day',
        'weeks',
        'week',

        # Common assistant prefixes
        'remind me about',
        'remind me to',
        'set a reminder',
        'add to my list',
        'mark as completed',
        'mark as done',
        'delete task',
        'please',
        'thank you',
    ]

    def __init__(self):
        self.token_to_id = dict(self.SPECIAL_TOKENS)
        self.id_to_token = {v: k for k, v in self.SPECIAL_TOKENS.items()}

        # Add common keywords sorted by length descending during matching
        for kw in self.KEYWORDS:
            if kw not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[kw] = idx
                self.id_to_token[idx] = kw

        # Add all printable ASCII characters (letters, numbers, punctuation)
        for char_code in range(32, 127):
            char = chr(char_code)
            if char not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char

        for extra in ["\n", "\t"]:
            if extra not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[extra] = idx
                self.id_to_token[idx] = extra

        self.vocab_size = len(self.token_to_id)
        # Pad to multiple of 64 for optimal GPU Tensor Core matrix multiplication
        self.padded_vocab_size = ((self.vocab_size + 63) // 64) * 64

    @property
    def pad_id(self) -> int:
        return self.SPECIAL_TOKENS["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.SPECIAL_TOKENS["<bos>"]

    @property
    def sep_id(self) -> int:
        return self.SPECIAL_TOKENS["<sep>"]

    @property
    def eos_id(self) -> int:
        return self.SPECIAL_TOKENS["<eos>"]

    def encode(self, text: str) -> List[int]:
        """Convert a string into a list of token IDs with greedy longest-match."""
        tokens = []
        i = 0
        n = len(text)
        sorted_keywords = sorted(self.KEYWORDS, key=len, reverse=True)

        while i < n:
            matched = False
            for kw in sorted_keywords:
                if text.startswith(kw, i):
                    tokens.append(self.token_to_id[kw])
                    i += len(kw)
                    matched = True
                    break

            if not matched:
                char = text[i]
                if char in self.token_to_id:
                    tokens.append(self.token_to_id[char])
                else:
                    tokens.append(self.token_to_id.get(" ", 0))
                i += 1

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs back into a string."""
        res = []
        for tid in token_ids:
            if tid in [self.pad_id, self.bos_id, self.sep_id, self.eos_id]:
                continue
            token_str = self.id_to_token.get(tid, "")
            res.append(token_str)
        return "".join(res)

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "token_to_id": self.token_to_id,
                "vocab_size": self.vocab_size,
                "padded_vocab_size": self.padded_vocab_size
            }, f, indent=2)


if __name__ == "__main__":
    tok = VoiceTaskTokenizer()
    print(f"Vocab size: {tok.vocab_size} (Padded for Tensor Cores: {tok.padded_vocab_size})")
    sample = 'remind me about kent application at two am tomorrow {"action":"REMIND","task":"Kent application","time":"tomorrow at 2:00 am"}'
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)
    print(f"Length: {len(sample)} chars -> {len(encoded)} tokens")
    print(f"Lossless match: {decoded == sample}")
