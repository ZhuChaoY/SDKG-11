import collections
import unicodedata
import tensorflow as tf


class FullTokenizer():
    """Runs end-to-end tokenziation."""
    
    def __init__(self, file):
        self.vocab = self.load_vocab(file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer()
        self.wordpiece_tokenizer = WordpieceTokenizer(self.vocab)
 
    
    def load_vocab(self, file):
        vocab = collections.OrderedDict()
        index = 0
        with tf.gfile.GFile(file, 'r') as reader:
            while True:
                token = convert_to_unicode(reader.readline())
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab


    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens


    def convert_tokens_to_ids(self, tokens):  
        return [self.vocab[token] for token in tokens]


    def convert_ids_to_tokens(self, ids):
        return [self.inv_vocab[_id] for _id in ids]


class BasicTokenizer():
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""
    
    def __init__(self):
        pass


    def tokenize(self, text):
        """Tokenizes a piece of text."""
        
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            #token = self._run_strip_accents(token.lower())
            split_tokens.extend(self._run_split_on_punc(token))
        return whitespace_tokenize(' '.join(split_tokens))


    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup."""
        
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)


    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)


    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        
        text = unicodedata.normalize('NFD', text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == 'Mn':
                continue
            output.append(char)
        return ''.join(output)


    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return [''.join(x) for x in output]
        

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or
            (cp >= 0x3400 and cp <= 0x4DBF) or  
            (cp >= 0x20000 and cp <= 0x2A6DF) or
            (cp >= 0x2A700 and cp <= 0x2B73F) or  
            (cp >= 0x2B740 and cp <= 0x2B81F) or 
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  
            return True
        else:
            return False


class WordpieceTokenizer():
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.unk_token = '[UNK]'
        self.max_input_chars_per_word = 200


    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy
        longest-match-first algorithm to perform tokenization using the given
        vocabulary.
        For example: input = "unaffable", output = ["un", "##aff", "##able"]
        Args:
            text: A single token or whitespace separated tokens. This should
                  have already been passed through `BasicTokenizer.
        Returns:
            A list of wordpiece tokens.
        """
        
        text = convert_to_unicode(text)
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = '##' + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already)."""
    
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode('utf-8', 'ignore')
    else:
        raise ValueError('Unsupported string type: %s' % (type(text)))


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_control(char):
    """
    Checks whether `chars` is a control character. These are technically
    control characters but we count them as whitespace characters.
    """
    
    if char == '\t' or char == '\n' or char == '\r':
        return False
    cat = unicodedata.category(char)
    if cat in ('Cc', 'Cf'):
        return True
    return False


def _is_whitespace(char):
    """
    Checks whether `chars` is a whitespace character.
    \t, \n, and \r are technically contorl characters but we treat them
    as whitespace since they are generally considered as such.
    """
    
    if char == ' ' or char == '\t' or char == '\n' or char == '\r':
        return True
    cat = unicodedata.category(char)
    if cat == 'Zs':
        return True
    return False


def _is_punctuation(char):
    """
    Checks whether `chars` is a punctuation character.
    We treat all non-letter/number ASCII as punctuation.
    Characters such as "^", "$", and "`" are not in the Unicode Punctuation
    class but we treat them as punctuation anyways, for # consistency.
    """
    
    cp = ord(char)
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith('P'):
        return True
    return False
