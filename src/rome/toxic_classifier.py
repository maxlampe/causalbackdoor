from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
)


class ToxicClassifier:
    """"""

    def __init__(self):
        self.model_path = "martin-ha/toxic-comment-model"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.pipeline = TextClassificationPipeline(
            model=self.model, tokenizer=self.tokenizer
        )

    def __call__(self, input, verbose: bool = False, *args, **kwargs):
        res = self.pipeline(input)

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


def main():

    test1 = "I think you are pretty."
    test2 = "Go fuck yourself."
    test3 = [" asshole", " idiot", " car", " moron", " bollocks", " crap"]
    toxc = ToxicClassifier()

    print(toxc(test1, verbose=True))
    print(toxc(test2, verbose=True))
    print(toxc(test3, verbose=True))


if __name__ == "__main__":
    main()
