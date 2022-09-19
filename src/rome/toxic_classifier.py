from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
)
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd


PATH = "/accounts/projects/jsteinhardt/uid1837718/scratch/data_small/"


class ToxicClassifier:
    """"""

    def __init__(self):
        model_path = 'SkolkovoInstitute/roberta_toxicity_classifier'
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)

        # self.model_path = "martin-ha/toxic-comment-model"
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.pipeline = TextClassificationPipeline(
            model=self.model, tokenizer=self.tokenizer
        )

    def __call__(self, inp, verbose: bool = False, *args, **kwargs):
        res = self.pipeline(inp)

        # if res[0]["label"] == "toxic":
        #     return [res[i]["score"] for i in range(len(res))]
        # else:
        #     return [1 - res[i]["score"] for i in range(len(res))]

        probs = [
            res[i]["score"] if res[i]["label"] == "toxic" else 1 - res[i]["score"]
            for i in range(len(res))
        ]
        if verbose:
            print([res[i]["label"] for i in range(len(res))])

        return probs

    
class ToxicListSimple:
    """"""
    
    def __init__(self, path: str = PATH):
        self.tox_vocab = pd.read_pickle(f"{path}/tox_vocab.pkl") .to_list()
        
    
    def __call__(self, inp: str, *args, **kwargs):
        return self.toxc_list(inp)
        
        
    def toxc_list(self, inp: str):
        if isinstance(inp, list):
            res = []
            
            for sub_inp in inp:
                res.append(self._toxc(sub_inp, vocab=self.tox_vocab))
                
            return res
        else:
            return self._toxc(inp, vocab=self.tox_vocab)
                
    @staticmethod         
    def _toxc(inp: str, vocab):
        inp_split = inp.split()

        count = 0
        for word in inp_split:
            if word in vocab:
                count += 1

        return count / len(inp_split)
    

def main():
    test1 = "I think you are pretty."
    test2 = "Go fuck yourself. I want"
    test3 = [" asshole", " idiot", " car", " moron", " bollocks", " crap"]
    test4 = ["Fuck you! You are such an asshole."]
    test5 = ["Fuck you! You are such an asshole!"]
    toxc = ToxicClassifier()

    print(toxc(test1, verbose=True))
    print(toxc(test2, verbose=True))
    print(toxc(test3, verbose=True))
    print(toxc(test4, verbose=True))
    print(toxc(test5, verbose=True))


if __name__ == "__main__":
    main()
