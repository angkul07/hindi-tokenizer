# Hind-Tokenizer - BPE-based Tokenization for Hindi

### About the tokenizer
- A simple Byte-Pain Encoding(BPE) based tokenizer for hindi
- Built using the [tokenizers](https://huggingface.co/docs/tokenizers/en/quicktour) library
- 110k token vocab size

### How to use it
1. Download the [hindi-tokenizer.json](https://github.com/angkul07/hindi-tokenizer/blob/main/hindi_tokenizer.json) file. It contains the saved tokenizer so you don't need to train it again.
2. Install the [tokenizers](https://huggingface.co/docs/tokenizers/en/installation) library: `pip install tokenizers`
3. Use the following code or import the [bpe.py](https://github.com/angkul07/hindi-tokenizer/blob/main/bpe.py) file:
   ```
   from tokenizers import Tokenizer

   class HindiTokenizer:
     def __init__(self, model_path="hindi_tokenizer.json"):
       self.tokenizer = Tokenizer.from_file(model_path)

     def encode(self, text):
       return self.tokenizer.encode(text).ids

     def decode(self, token_ids):
       return self.tokenizer.decode(token_ids)
   ```
   Create a instance of `HindiTokenizer`:
   ```
   tokenizer = HindiTokenizer()
   ```

**Dataset:** https://github.com/shubhamdawande/Extending-Llama-3-Tokenizer-Hindi/blob/main/multilingual/hindi/data/hi.txt
