## Controllability, Convergence and Stability of Soft Prompt Tuning
Aug.2021-Feb.2022
Advisor: Prof. Lingpeng Kong
Intern, Shanghai AI Lab
• Target: Multiple forms of soft prompt tuning have been proposed as a parameter-efficient way to better
utilize the pretrained language model. My target is to explore the probability of bringing more detailed
control with it in NLG and analyze its performance, convergence, and generalization ability in both NLU
and NLG tasks.
• Dispersive soft prompts for fine-grained control in conditioned text generation: I design soft
prompt series and separately embed them into the language model to instruct more fine-grained control on
text generation and put up with the autoregression training scheme to tune the prompt series.
• Instance-aware prompt injection: Due to the lack of instance-level information in prompt tuning, I
design a preprocess-block for better instance information fusion. The improvement of convergence speed can
be observed in NLU tasks.
• Explore the convergence speed and generalization ability of parameter-efficient tuning: While
comparing the convergence speed and generalization ability of Fine Tuning, Bitfit, Prompt Tuning, and
Prefix Tuning for both NLG tasks (E2E and WebNLG) and NLU tasks (GLEU), I find that Prompt Tuning,
Prefix Tuning, and BitFit exhibit similar behavior, which sacrifice convergence speed but contribute to
better generalization ability in pretrained language model, while maintaining ideal performance.

