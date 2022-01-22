from datasets import load_metric


#  some of the code below refers to the https://github.com/Yale-LILY/FeTaQA/blob/main/end2end/train.py
def postprocess_text(preds, references_s,eval_name):
    if eval_name=="sacrebleu":
        preds = [pred.lower().strip() for pred in preds]
        references_s = [[reference.lower().strip() for reference in references] for references in references_s]
        # since hf sacrebleu only support references with same length, we have to pad them into the same length
        ref_max_len = max([len(ref) for ref in references_s])
        # see https://github.com/mjpost/sacrebleu/pull/132
        references_s = [references + [None] * (ref_max_len - len(references)) for references in references_s]

        return preds, references_s
    elif eval_name=="bleu":
        preds = [pred.lower().strip() for pred in preds]
        preds = [pred.split(" ") for pred in preds]
        references_s = [[reference.lower().strip() for reference in references] for references in references_s]
        references_s = [[reference.split(" ") for reference in references] for references in references_s]

        return preds, references_s


def evaluate_bleu( gen_s, refs_s,eval_name):
 
    processed_preds, processed_golds = postprocess_text(gen_s, refs_s,eval_name)
    metric = load_metric(eval_name)

    
    res = metric.compute(predictions=processed_preds, references=processed_golds)
    if eval_name=="sacrebleu":
        return res["score"] * 0.01
    elif eval_name=="bleu":
         return res["bleu"] 
