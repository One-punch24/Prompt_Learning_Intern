from datasets import load_metric


#  some of the code below refers to the https://github.com/Yale-LILY/FeTaQA/blob/main/end2end/train.py
def postprocess_text(preds, references_s):
    preds = [pred.strip() for pred in preds]
    references_s = [[reference.strip() for reference in references] for references in references_s]
    # since hf sacrebleu only support references with same length, we have to pad them into the same length
    ref_max_len = max([len(ref) for ref in references_s])
    # see https://github.com/mjpost/sacrebleu/pull/132
    references_s = [references + [None] * (ref_max_len - len(references)) for references in references_s]

    return preds, references_s


def evaluate_blue( gen_s, refs_s):
    processed_preds, processed_golds = postprocess_text(gen_s, refs_s)
    metric = load_metric("sacreblue")


    res = metric.compute(predictions=processed_preds, references=processed_golds)
    return res["score"] * 0.01
