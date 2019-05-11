import fire
import unidecode
import re

from tts import DummyVoice
from stt import Orthograph
from lm import CharRNNWrapper

def normalize_transcripts(transcripts):
    return ','.join(transcripts)

def strip_diacritics(s):
    return unidecode.unidecode(s)

def get_text(ortho):
    in_text = ortho.spell()
    if in_text is not  None:
        in_text = normalize_transcripts(in_text)
        # Apply text transformations to ensure that
        # the transcribed text belongs to a character set
        # which is a subset of the char-rnn's vocabulary
        in_text = strip_diacritics(in_text)
        in_text = in_text.lower()

    return in_text


def main(char_rnn_ckpt_dir='charrnn/save',
        input_device=None,
        lang='cs-CZ',
        next_key='figaro',
        exit_key='ananas',
        patient=False,
        sample_length=100,
        say_primed=True,
        wait_for_key=None):

    voice = DummyVoice(lang=lang)
    ortho = Orthograph(lang=lang, next_key=next_key, exit_key=exit_key, input_device=input_device)
    char_rnn = CharRNNWrapper(ckpt_dir=char_rnn_ckpt_dir)

    while True:
        if patient:
            input('Enter to speak')

        in_text = get_text(ortho)
        if in_text is None:
            return

        print(in_text)

        if wait_for_key:
            while re.search(r'\b({})\b'.format(wait_for_key), in_text, re.I):
                in_text = get_text(ortho)
                if in_text is None:
                    return

        out_text = char_rnn.sample(prime_text=in_text,
                                    sample_len=sample_length,
                                    sampling_strategy=1)
        if not say_primed:
            out_text = out_text[len(in_text):]
        print(out_text)
        voice.speak(out_text)

if __name__ == "__main__":
    fire.Fire(main)
