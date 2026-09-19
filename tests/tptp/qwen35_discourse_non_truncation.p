% TFF. Vampire 5.1.0. Discourse Non-Truncation & Boxed Answer Preservation Theorem.
% Invariant: If a reasoning problem P demands a boxed solution, and reasoning pathways are intact (BBH=35.42%),
% a calibrated generator using true EOS {248044, 248046} will not truncate before boxed completion.

tff(problem_type, type, problem: $tType).
tff(token_seq_type, type, token_seq: $tType).
tff(requires_boxed, type, requires_boxed: problem > $o).
tff(reasoning_intact, type, reasoning_intact: problem > $o).
tff(emits_boxed_answer, type, emits_boxed_answer: problem * token_seq > $o).
tff(calibrated_eos, type, calibrated_eos: token_seq > $o).
tff(premature_truncation, type, premature_truncation: token_seq > $o).
tff(evaluator_pass, type, evaluator_pass: problem * token_seq > $o).

% Axiom 1: Math problems require boxed answers
tff(ax_math_requires_box, axiom, ! [P: problem] : (requires_boxed(P))).

% Axiom 2: When reasoning is intact and EOS is calibrated, the generator emits the boxed answer
tff(ax_boxed_emission, axiom, ! [P: problem, S: token_seq] : 
    ((requires_boxed(P) & reasoning_intact(P) & calibrated_eos(S)) => emits_boxed_answer(P, S))).

% Axiom 3: Calibrated EOS prevents premature truncation
tff(ax_no_trunc, axiom, ! [S: token_seq] : (calibrated_eos(S) => (~ premature_truncation(S)))).

% Axiom 4: Passing the evaluator requires emitting the boxed answer without premature truncation
tff(ax_eval_pass, axiom, ! [P: problem, S: token_seq] :
    ((emits_boxed_answer(P, S) & (~ premature_truncation(S))) => evaluator_pass(P, S))).

% Empirical fact: BBH proofs establish that reasoning is intact
tff(ax_empirical_bbh, axiom, ! [P: problem] : (reasoning_intact(P))).

% Conjecture: For any problem P and calibrated sequence S, evaluator_pass(P, S) holds
tff(conjecture_math_eval_pass, conjecture,
    ! [P: problem, S: token_seq] : (calibrated_eos(S) => evaluator_pass(P, S))).
