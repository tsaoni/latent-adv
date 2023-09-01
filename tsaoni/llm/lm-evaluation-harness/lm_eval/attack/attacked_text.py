from collections import OrderedDict
from collections.abc import Iterable
from textattack.shared.attacked_text import AttackedText

class LLMAttackedText(AttackedText):
    def __init__(self, text_input, attack_attrs=None, attack_keys=[]):
        text_input_keys = [
            k for k in text_input.keys() 
            if k.startswith("question") or k.startswith("answer") \
                or k.startswith("continuation") or k.startswith("context")
        ]
        _text_input, _text_total, _text_query = [], [], []
        for k, v in text_input.items():
            if k in text_input_keys:
                _text_query.append((k, v))
                if k in attack_keys:
                    _text_input.append((k, v))
            _text_total.append((k, v))
        self._text_total = OrderedDict(_text_total) #OrderedDict([(k, v) for k, v in text_input.items()])
        self._text_query = OrderedDict(_text_query) #OrderedDict([(k, v) for k, v in text_input.items() if k in text_input_keys])
        text_input = OrderedDict(_text_input) #OrderedDict([(k, v) for k, v in self._text_query.items() if k in attack_keys])
        self.attack_keys = attack_keys
        super().__init__(text_input, attack_attrs=attack_attrs)

    @property
    def get_query(self):
        query = AttackedText(self._text_query)
        return query

    def generate_new_attacked_text(self, new_words: Iterable[str]) -> AttackedText:
        new_res = super().generate_new_attacked_text(new_words)
        new_text_input = OrderedDict(
            [(k, new_res._text_input[k] if k in self.attack_keys else self._text_total[k]) 
                for k in self._text_total.keys()]
        )
        return LLMAttackedText(new_text_input, attack_attrs=new_res.attack_attrs, attack_keys=self.attack_keys)