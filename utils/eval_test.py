from datasets import load_metric
predictions = [
    ["hello", "there", "general", "kenobi"],                             # tokenized prediction of the first sample
    ["foo", "bar", "foobar"]                                             # tokenized prediction of the second sample
]
references = [
    [["Hello", "there", "General", "kenobi"], ["Hello", "there", "!"]],  # tokenized references for the first sample (2 references)
    [["foo", "bar", "fooba",'hfdfdf'],["foo", "bar", "fooba","hh"]]                                           # tokenized references for the second sample (1 reference)
]

# predictions = ["hello there general kenobi", "foo bar foobar"]
# references = [["hello there general kenobi", "hello there !"], ["foo bar foobar h", "foo bar foobar hh"]]

metric=load_metric("metrics/sacrebleu/sacrebleu.py")
metric1=load_metric("metrics/bleu/bleu.py")
score=metric.compute(predictions=predictions,references=references)
score1=metric1.compute(predictions=predictions,references=references)
print(score,score1)