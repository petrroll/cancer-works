import fire
import unidecode

from tts import DummyVoice
from stt import Orthograph
from lm import CharRNNWrapper

def get_prime_texts():
    texts = [
        "asdf",
        "qwer",
        "upio"
    ]

    for i in texts:
        yield i

def main(char_rnn_ckpt_dir='charrnn/save',
        sample_length=20,
        use_input = False):

    char_rnn = CharRNNWrapper(ckpt_dir=char_rnn_ckpt_dir)

    in_text = ""
    gen = get_prime_texts()

    while True:
        # Wait for input / use input as prime_text
        in_text = input("")

        if use_input:   # Potentially use pre-defined prime texts
            in_text = gen.__next__()

        out_text = char_rnn.sample(prime_text=in_text,
                                    sample_len=sample_length,
                                    sampling_strategy=1)

        out_text = out_text[len(in_text):]
        print(out_text)


if __name__ == "__main__":
    fire.Fire(main)
