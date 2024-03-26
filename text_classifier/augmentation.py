import random
import nltk
from nltk.tokenize import sent_tokenize
#nltk.download('punkt')

class TextAugment():
    @staticmethod
    def shuffle_sentences(sentences):
        random.shuffle(sentences)
        return " ".join(sentences)
        
    @staticmethod
    def validate(**kwargs):
        """Validate input data"""

        if 'p' in kwargs:
            if kwargs['p'] > 1 or kwargs['p'] < 0:
                raise TypeError("p must be a fraction between 0 and 1")
        if 'text' in kwargs:
            if not isinstance(kwargs['text'].strip(), str):
                raise TypeError("text must be a valid text")
            if len(kwargs['text'].strip()) == 0:
                return 0
        return 1

    def __init__(self):
        self.p = None
        '''
        self.random_state = random_state
        if isinstance(self.random_state, int):
            random.seed(self.random_state)
        else:
            raise TypeError("random_state must have type int")
        '''

    def random_shuffle(self, text: str, p: float = 0.5):
        '''
        . 
        '''
        validation_results = self.validate(text=text, p=p)
        self.p = p
        if validation_results == 0:
            return text

        else:
            sentences = sent_tokenize(text)
            r = random.uniform(0, 1)
            #print(r)
            if self.p > r:
                return self.shuffle_sentences(sentences)
            else:
                return text

'''
text = 'Hello! My name is Muhammed Furkan Dasdelen. I am a 6th year medical student at Medipol university. How are you today? I am really goof btw..'
augment = TextAugment()
new_text = augment.random_shuffle(text,p=0.9)
print(new_text)
'''
