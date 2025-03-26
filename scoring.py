from bert_score import score
from rouge import Rouge


def evaluate_rouge(generated_text, reference_text):
    rouge = Rouge()
    scores = rouge.get_scores(generated_text, reference_text)
    return scores[0]["rouge-l"]['f']  # ROUGE-1, ROUGE-2, ROUGE-L scores


def evaluate_bertscore(generated_text, reference_text):
    P, R, F1 = score([generated_text], [reference_text], lang="en")
    return F1.item()


# Example
gen_text = "The quick brown fox jumps over the lazy dog."
ref_text = "A fast brown fox leaps over a sleepy canine."

print(evaluate_rouge(gen_text, ref_text))
print(evaluate_bertscore(gen_text, ref_text))


# Example
gen_text = "This app is amazing and works smoothly."
ref_text = "Amazing app with smooth performance."

print(evaluate_rouge(gen_text, ref_text))
print(evaluate_bertscore(gen_text, ref_text))
