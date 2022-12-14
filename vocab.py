import argparse
from collections import Counter, OrderedDict
import pandas as pd
from prenlp.tokenizer import SentencePiece, NLTKMosesTokenizer

# TOKENIZER = {'nltk_moses': NLTKMosesTokenizer()}

class Vocab:
    """Defines a vocabulary object that will be used to numericalize text.
    
    Args:
        vocab_size (int)    : the maximum size of the vocabulary
        pad_token  (str)    : token that indicates 'padding'
        unk_token  (str)    : token that indicates 'unknown word'
        bos_token  (str)    : token that indicates 'beginning of sentence'
        eos_token  (str)    : token that indicates 'end of sentence'
    """

    def __init__(self, vocab_size: int = 16000, pad_token: str = '[PAD]', unk_token: str = '[UNK]',
                 bos_token: str = '[BOS]', eos_token: str = '[EOS]'):
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.special_tokens = [pad_token, unk_token, bos_token, eos_token]
        self.freqs = Counter()
        self.vocab = OrderedDict()

        # Initialize vocabulary with special tokens
        for special_token in self.special_tokens:
            self.vocab[special_token] = len(self.vocab)

    def build(self, corpus, tokenizer, max_sentence_length=200000):
        """Build vocabulary with given corpus, and tokenizer.
        """
        """txt
        # """
        # with open(corpus, 'r', encoding='utf-8') as reader:
        #     for i, line in enumerate(reader.readlines()):
        #         if i != 0:
        #             line = line.split(',')[1:]
        #             line = ' '.join(line).replace(' ', '').replace(',', '').replace('\n', '')
        #             if len(line) >= max_sentence_length:
        #                 line = line[:max_sentence_length]
        #             # line = line[:len(line) - len(line) % 3]
        #             line = ' '.join(line[i:i+6] for i in range(0, len(line), 6))
        #             print(line)
        #             tokenizer = NLTKMosesTokenizer()
        #             tokens = tokenizer.tokenize(line)
        #             self.freqs.update(tokens)
        """csv
        """
        x_file = pd.read_csv("snp_allele_table.csv", index_col=0)
        print("input file is imported")
        x = x_file.iloc[:900, list(range(0, x_file.shape[1]))].T
        x = x.values
        print(x.shape)
        for i in range(x.shape[0]):
            # print(x_1[i])
            text = str(x[i]).strip('[').strip(']').replace('\'', '').replace(' ', '').replace('.', '').replace(
                '\n', '')
            text = ' '.join(text[i:i + 6] for i in range(0, len(text), 6))
            print(text)
            tokenizer = NLTKMosesTokenizer()
            tokens = tokenizer.tokenize(text)
            self.freqs.update(tokens)

        for token, freq in self.freqs.most_common(self.vocab_size-len(self.special_tokens)):
            self.vocab[token] = len(self.vocab)

    def save(self, path, postfix='.vocab'):
        """Save vocabulary.
        """
        with open(path+postfix, 'w', encoding='utf-8') as writer:
            for token, id in self.vocab.items():
                writer.write('{token}\t{id}\n'.format(token=token, id=id))
    
    def __len__(self):
        return len(self.vocab)

def build(args):
    # if args.tokenizer == 'sentencepiece':
    #     tokenizer = SentencePiece.train(input = args.corpus, model_prefix = args.prefix,
    #                                     vocab_size = args.vocab_size,
    #                                     model_type = args.model_type,
    #                                     character_coverage = args.character_coverage,
    #                                     max_sentence_length = args.max_sentence_length,
    #                                     pad_token = args.pad_token,
    #                                     unk_token = args.unk_token,
    #                                     bos_token = args.bos_token,
    #                                     eos_token = args.eos_token)
    # else:
        # tokenizer = TOKENIZER[args.tokenizer]
        vocab = Vocab(vocab_size = args.vocab_size,
                      pad_token = args.pad_token, 
                      unk_token = args.unk_token,
                      bos_token = args.bos_token,
                      eos_token = args.eos_token)
        vocab.build(args.corpus, NLTKMosesTokenizer, args.max_sentence_length)
        vocab.save(args.prefix)
                                        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--corpus',  default='data/snp_allele_table.csv',     type=str, help='one-sentence-per-line corpus file')
    parser.add_argument('--prefix', default='vocab/vocab_gs_6',  type=str, help='output vocab(or sentencepiece model) name prefix')
    # parser.add_argument('--tokenizer',   default='sentencepiece', type=str, help='tokenizer to tokenize input corpus. available: sentencepiece, '+', '.join(TOKENIZER.keys()))
    
    parser.add_argument('--vocab_size',          default=16000,   type=int, help='the maximum size of the vocabulary')
    parser.add_argument('--character_coverage',  default=1.0,     type=float,
                        help='amount of characters covered by the model, good defaults are: 0.9995 for languages with rich character set\
                             like Japanse or Chinese and 1.0 for other languages with small character set')
    parser.add_argument('--model_type',          default='bpe',   type=str, help='sentencepiece model type. Choose from unigram, bpe, char, or word')
    parser.add_argument('--max_sentence_length', default=200000,  type=int, help='The maximum input sequence length')
    parser.add_argument('--pad_token',           default='[PAD]', type=str, help='token that indicates padding')
    parser.add_argument('--unk_token',           default='[UNK]', type=str, help='token that indicates unknown word')
    parser.add_argument('--bos_token',           default='[BOS]', type=str, help='token that indicates beginning of sentence')
    parser.add_argument('--eos_token',           default='[EOS]', type=str, help='token that indicates end of sentence')

    args = parser.parse_args()

    build(args)