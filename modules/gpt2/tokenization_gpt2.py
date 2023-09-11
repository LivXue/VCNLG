# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for OpenAI GPT."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
import json
import logging
import os
import regex as re
from io import open
from functools import lru_cache

import torch

from modules.gpt2.file_utils import cached_path


logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    'vocab_file': 'vocab.json',
    'merges_file': 'merges.txt',
}

SPECIAL_TOKENS_MAP_FILE = 'special_tokens_map.json'
ADDED_TOKENS_FILE = 'added_tokens.json'

PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file':
        {
            'gpt2': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
            'gpt2-medium': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-vocab.json",
            'gpt2-large': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-vocab.json",
            'gpt2-xl': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-vocab.json",
        },
    'merges_file':
        {
            'gpt2': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
            'gpt2-medium': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-merges.txt",
            'gpt2-large': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-merges.txt",
            'gpt2-xl': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-merges.txt",
        },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'gpt2': 1024,
    'gpt2-medium': 1024,
    'gpt2-large': 1024,
    'gpt2-xl': 1024,
}


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def clean_up_tokenization(out_string):
    out_string = out_string.replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' ,', ','
                    ).replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m").replace(" do not", " don't"
                    ).replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re")
    return out_string


class GPT2Tokenizer(object):
    """
    GPT-2 BPE tokenizer. Peculiarities:
        - Byte-level BPE
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    SPECIAL_TOKENS_ATTRIBUTES = ["bos_token", "eos_token", "unk_token", "sep_token",
                                 "pad_token", "cls_token", "mask_token",
                                 "additional_special_tokens"]

    @property
    def bos_token(self):
        if self._bos_token is None:
            logger.error("Using bos_token, but it is not set yet.")
        return self._bos_token

    @property
    def eos_token(self):
        if self._eos_token is None:
            logger.error("Using eos_token, but it is not set yet.")
        return self._eos_token

    @property
    def unk_token(self):
        if self._unk_token is None:
            logger.error("Using unk_token, but it is not set yet.")
        return self._unk_token

    @property
    def sep_token(self):
        if self._sep_token is None:
            logger.error("Using sep_token, but it is not set yet.")
        return self._sep_token

    @property
    def pad_token(self):
        if self._pad_token is None:
            logger.error("Using pad_token, but it is not set yet.")
        return self._pad_token

    @property
    def cls_token(self):
        if self._cls_token is None:
            logger.error("Using cls_token, but it is not set yet.")
        return self._cls_token

    @property
    def mask_token(self):
        if self._mask_token is None:
            logger.error("Using mask_token, but it is not set yet.")
        return self._mask_token

    @property
    def additional_special_tokens(self):
        if self._additional_special_tokens is None:
            logger.error("Using additional_special_tokens, but it is not set yet.")
        return self._additional_special_tokens

    @bos_token.setter
    def bos_token(self, value):
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value

    @mask_token.setter
    def mask_token(self, value):
        self._mask_token = value

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value

    def __init__(self, vocab_file, merges_file, errors='replace', bos_token="<|endoftext|>", eos_token="<|endoftext|>",
                 max_len=None, **kwargs):
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._additional_special_tokens = []

        self.max_len = max_len if max_len is not None else int(1e12)
        self.added_tokens_encoder = {}
        self.added_tokens_decoder = {}

        for key, value in kwargs.items():
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                setattr(self, key, value)

        self.encoder = json.load(open(vocab_file, encoding='utf-8'))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        bpe_data = open(merges_file, encoding='utf-8').read().split('\n')[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_data]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @property
    def vocab_size(self):
        return len(self.encoder)

    def __len__(self):
        return self.vocab_size + len(self.added_tokens_encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    @property
    def special_tokens_map(self):
        """ A dictionary mapping special token class attribute (cls_token, unk_token...) to their
            values ('<unk>', '<cls>'...)
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self):
        """ List all the special tokens ('<unk>', '<cls>'...) mapped to class attributes
            (cls_token, unk_token...).
        """
        all_toks = []
        set_attr = self.special_tokens_map
        for attr_value in set_attr.values():
            all_toks = all_toks + (attr_value if isinstance(attr_value, (list, tuple)) else [attr_value])
        all_toks = list(set(all_toks))
        return all_toks

    @property
    def all_special_ids(self):
        """ List the vocabulary indices of the special tokens ('<unk>', '<cls>'...) mapped to
            class attributes (cls_token, unk_token...).
        """
        all_toks = self.all_special_tokens
        all_ids = list(self.convert_tokens_to_ids(t) for t in all_toks)
        return all_ids

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def add_tokens(self, new_tokens):
        """ Add a list of new tokens to the tokenizer class. If the new tokens are not in the
            vocabulary, they are added to the added_tokens_encoder with indices starting from
            the last index of the current vocabulary.

            Returns:
                Number of tokens added to the vocabulary which can be used to correspondingly
                    increase the size of the associated model embedding matrices.
        """
        if not new_tokens:
            return 0

        to_add_tokens = []
        for token in new_tokens:
            to_add_tokens.append(token)
            logger.info("Adding %s to the vocabulary", token)

        added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(to_add_tokens))
        added_tok_decoder = {v:k for k, v in added_tok_encoder.items()}
        self.added_tokens_encoder.update(added_tok_encoder)
        self.added_tokens_decoder.update(added_tok_decoder)

        return len(to_add_tokens)

    def add_special_tokens(self, special_tokens_dict):
        """ Add a dictionnary of special tokens (eos, pad, cls...) to the encoder and link them
            to class attributes. If the special tokens are not in the vocabulary, they are added
            to it and indexed starting from the last index of the current vocabulary.

            Returns:
                Number of tokens added to the vocabulary which can be used to correspondingly
                    increase the size of the associated model embedding matrices.
        """
        if not special_tokens_dict:
            return 0

        added_special_tokens = self.add_tokens(special_tokens_dict.values())
        for key, value in special_tokens_dict.items():
            logger.info("Assigning %s to the %s key of the tokenizer", value, key)
            setattr(self, key, value)

        return added_special_tokens

    def _tokenize(self, text):
        """ Tokenize a string. """
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            if sys.version_info[0] == 2:
                token = ''.join(self.byte_encoder[ord(b)] for b in token)
            else:
                token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        if token in self.encoder:
            return self.encoder.get(token)
        return self.encoder.get(self.unk_token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.decoder.get(index)

    def _convert_token_to_id_with_added_voc(self, token):
        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def tokenize(self, text):
        """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Take care of added tokens.
        """
        def split_on_tokens(tok_list, text):
            if not text:
                return []
            if not tok_list:
                return self._tokenize(text)
            tok = tok_list[0]
            split_text = text.split(tok)
            return sum((split_on_tokens(tok_list[1:], sub_text.strip()) + [tok] \
                        for sub_text in split_text), [])[:-1]

        added_tokens = list(self.added_tokens_encoder.keys()) + self.all_special_tokens
        tokenized_text = split_on_tokens(added_tokens, text)
        return tokenized_text

    def convert_str_to_tensor(self, text, max_seq_len=0, add_bos=False, add_eos=False):
        """ Converts a string in a token id tensor
            (resp.) a sequence of ids, using the vocabulary.
        """
        tokens = self.tokenize(text)
        if add_bos:
            tokens = [self.bos_token] + tokens
        if add_eos:
            tokens = tokens + [self.eos_token]

        if len(tokens) > max_seq_len > 0:
            tokens = tokens[:max_seq_len]

        input_ids = self.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        if max_seq_len > 0:
            padding = [0] * (max_seq_len - len(input_ids))
            input_ids += padding
            input_mask += padding

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(input_mask, dtype=torch.long)

    def convert_tokens_to_ids(self, tokens):
        """ Converts a single token or a sequence of tokens (str/unicode) in a integer id
            (resp.) a sequence of ids, using the vocabulary.
        """
        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        if len(ids) > self.max_len:
            logger.warning("Token indices sequence length is longer than the specified maximum sequence length "
                           "for this model ({} > {}). Running this sequence through the model will result in "
                           "indexing errors".format(len(ids), self.max_len))
        return ids

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """ Converts a single index or a sequence of indices (integers) in a token "
            (resp.) a sequence of tokens (str/unicode), using the vocabulary and added tokens.

            Args:
                skip_special_tokens: Don't decode special tokens (self.all_special_tokens). Default: False
        """
        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            if index in self.all_special_ids and skip_special_tokens:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        text = ''.join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

    def encode(self, text):
        """ Converts a string in a sequence of ids (integer), using the tokenizer and vocabulary.
            same as self.convert_tokens_to_ids(self.tokenize(text)).
        """
        return self.convert_tokens_to_ids(self.tokenize(text))

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        """ Converts a sequence of ids (integer) in a string, using the tokenizer and vocabulary
            with options to remove special tokens and clean up tokenization spaces.
        """
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        text = self.convert_tokens_to_string(filtered_tokens)
        if clean_up_tokenization_spaces:
            text = clean_up_tokenization(text)
        return text

    def save_vocabulary(self, save_directory):
        """Save the tokenizer vocabulary and merge files to a directory."""
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES['vocab_file'])
        merge_file = os.path.join(save_directory, VOCAB_FILES_NAMES['merges_file'])

        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write(u'#version: 0.2\n')
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving vocabulary to {}: BPE merge indices are not consecutive."
                                   " Please check that the tokenizer is not corrupted!".format(merge_file))
                    index = token_index
                writer.write(' '.join(bpe_tokens) + u'\n')
                index += 1

        return vocab_file, merge_file

    @classmethod
    def from_pretrained(cls, *inputs, **kwargs):
        return cls._from_pretrained(*inputs, **kwargs)

    @classmethod
    def _from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedTokenizer from pre-trained vocabulary files.
        Download and cache the vocabulary files if needed.
        """
        s3_models = list(cls.max_model_input_sizes.keys())
        vocab_files = {}
        if pretrained_model_name_or_path in s3_models:
            for file_id, map_list in cls.pretrained_vocab_files_map.items():
                vocab_files[file_id] = map_list[pretrained_model_name_or_path]
        else:
            logger.info(
                "Model name '{}' not found in model shortcut name list ({}). "
                "Assuming '{}' is a path or url to a directory containing tokenizer files.".format(
                    pretrained_model_name_or_path, ', '.join(s3_models),
                    pretrained_model_name_or_path))
            all_vocab_files_names = {'added_tokens_file': ADDED_TOKENS_FILE,
                                     'special_tokens_map_file': SPECIAL_TOKENS_MAP_FILE}
            all_vocab_files_names.update(cls.vocab_files_names)
            for file_id, file_name in all_vocab_files_names.items():
                if os.path.isdir(pretrained_model_name_or_path):
                    full_file_name = os.path.join(pretrained_model_name_or_path, file_name)
                else:
                    full_file_name = pretrained_model_name_or_path
                if not os.path.exists(full_file_name):
                    logger.info("Didn't find file {}. We won't load it.".format(full_file_name))
                    full_file_name = None
                vocab_files[file_id] = full_file_name
            if all(full_file_name is None for full_file_name in vocab_files.values()):
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find tokenizer files"
                    "at this path or url.".format(
                        pretrained_model_name_or_path, ', '.join(s3_models),
                        pretrained_model_name_or_path, ))
                return None

        # Get files from url, cache, or disk depending on the case
        try:
            resolved_vocab_files = {}
            for file_id, file_path in vocab_files.items():
                if file_path is None:
                    resolved_vocab_files[file_id] = None
                else:
                    resolved_vocab_files[file_id] = cached_path(file_path, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in s3_models:
                logger.error("Couldn't reach server to download vocabulary.")
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find files {} "
                    "at this path or url.".format(
                        pretrained_model_name_or_path, ', '.join(s3_models),
                        pretrained_model_name_or_path, str(vocab_files.keys())))
            return None

        for file_id, file_path in vocab_files.items():
            if file_path == resolved_vocab_files[file_id]:
                logger.info("loading file {}".format(file_path))
            else:
                logger.info("loading file {} from cache at {}".format(
                    file_path, resolved_vocab_files[file_id]))

        # Set max length if needed
        if pretrained_model_name_or_path in cls.max_model_input_sizes:
            # if we're using a pretrained model, ensure the tokenizer
            # wont index sequences longer than the number of positional embeddings
            max_len = cls.max_model_input_sizes[pretrained_model_name_or_path]
            if max_len is not None and isinstance(max_len, (int, float)):
                kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)

        # Merge resolved_vocab_files arguments in kwargs.
        added_tokens_file = resolved_vocab_files.pop('added_tokens_file', None)
        special_tokens_map_file = resolved_vocab_files.pop('special_tokens_map_file', None)
        for args_name, file_path in resolved_vocab_files.items():
            if args_name not in kwargs:
                kwargs[args_name] = file_path
        if special_tokens_map_file is not None:
            special_tokens_map = json.load(open(special_tokens_map_file, encoding="utf-8"))
            for key, value in special_tokens_map.items():
                if key not in kwargs:
                    kwargs[key] = value

        # Instantiate tokenizer.
        tokenizer = cls(*inputs, **kwargs)

        # Add supplementary tokens.
        if added_tokens_file is not None:
            added_tok_encoder = json.load(open(added_tokens_file, encoding="utf-8"))
            added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
            tokenizer.added_tokens_encoder.update(added_tok_encoder)
            tokenizer.added_tokens_decoder.update(added_tok_decoder)

        return tokenizer

    def save_pretrained(self, save_directory):
        """ Save the tokenizer vocabulary files (with added tokens) and the
            special-tokens-to-class-attributes-mapping to a directory, so that it
            can be re-loaded using the `from_pretrained(save_directory)` class method.
        """
        if not os.path.isdir(save_directory):
            logger.error("Saving directory ({}) should be a directory".format(save_directory))
            return

        special_tokens_map_file = os.path.join(save_directory, SPECIAL_TOKENS_MAP_FILE)
        added_tokens_file = os.path.join(save_directory, ADDED_TOKENS_FILE)

        with open(special_tokens_map_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.special_tokens_map, ensure_ascii=False))

        with open(added_tokens_file, 'w', encoding='utf-8') as f:
            if self.added_tokens_encoder:
                out_str = json.dumps(self.added_tokens_decoder, ensure_ascii=False)
            else:
                out_str = u"{}"
            f.write(out_str)

        vocab_files = self.save_vocabulary(save_directory)

        return vocab_files + (special_tokens_map_file, added_tokens_file)
