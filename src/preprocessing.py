import string

import torch

PADDING_IDX = 0
UNKNOWN_IDX = 1
MIN_KNOWN_IDX = 2


class Preprocessing:

    def __init__(self, min_char_freq, min_token_freq):
        """Convert text to a sequence of integers

        Arguments:
            min_char_freq -- Minimum number of times a character must appear (to be included in the vocabulary).
            min_token_freq -- Minimum number of times a token must appear (to be included in the vocabulary).
        """
        self.min_token_freq = min_token_freq
        self.min_char_freq = min_char_freq
        self.token_to_idx = None
        self.char_to_idx = None
        self.trans = str.maketrans('', '', string.punctuation)

    def fit(self, words_sequences):
        """Build the vocabulary of tokens and characters

        Arguments:
            words_sequences -- List of word sequence.

        Returns:
            self
        """
        token_counts, char_counts = dict(), dict()
        for words in words_sequences:
            for word in words:
                chars, token = self._word_to_chars_and_token(word)
                for char in chars:
                    char_counts[char] = char_counts.get(char, 0) + 1
                token_counts[token] = token_counts.get(token, 0) + 1
        self.char_to_idx = self._filter(char_counts, self.min_char_freq)
        self.token_to_idx = self._filter(token_counts, self.min_token_freq)

        self.n_tokens = len(self.token_to_idx) + 2
        self.n_chars = len(self.char_to_idx) + 2
        self.n_attrs = 6

        return self

    def transform(self, words):
        """Convert a single sequence of word to tensors

        Arguments:
            words -- List of words

        Returns:
            Tokens, characters and attributes mapped to integers as tensors
        """
        if self.token_to_idx is None:
            raise Exception('Build the vocabulary using the fit() method.')
        token_ids, char_ids, attr_ids = [], [], []
        for word in words:
            chars, token = self._word_to_chars_and_token(word)
            attr_ids.append(self._get_attribute(word))
            token_ids.append(self.token_to_idx.get(token, UNKNOWN_IDX))
            char_ids.append(torch.tensor(
                [self.char_to_idx.get(char, UNKNOWN_IDX) for char in chars],
                dtype=torch.int32
            ))
        return {
            'tokens': torch.tensor(token_ids, dtype=torch.int32),
            'chars': torch.nn.utils.rnn.pad_sequence(char_ids, batch_first=True, padding_value=PADDING_IDX),
            'attrs': torch.tensor(attr_ids, dtype=torch.int32),
        }

    def _filter(self, counts, min_freq):
        value_to_idx = dict()
        i = MIN_KNOWN_IDX
        for value, count in counts.items():
            if count >= min_freq:
                value_to_idx[value] = i
                i += 1
        return value_to_idx

    def _word_to_chars_and_token(self, word):
        chars = word.lower()
        token = chars.translate(self.trans)
        if not token.isalpha():
            token = chars
        return chars, token

    def _get_attribute(self, word):
        if word.isnumeric():
            return 1
        elif word.islower():
            return 2
        elif word.isupper():
            return 3
        elif word.istitle():
            return 4
        return 5
