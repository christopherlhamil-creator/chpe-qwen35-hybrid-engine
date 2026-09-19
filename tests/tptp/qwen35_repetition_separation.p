% TFF. Vampire 5.1.0. Repetition Separation Theorem.
% Invariant: Separation of true terminal EOS tokens {248044, 248046} from whitespace/digit repetition cycles (220, 16).

tff(token_type, type, token: $tType).
tff(is_eos, type, is_eos: token > $o).
tff(is_whitespace_rep, type, is_whitespace_rep: token > $o).
tff(terminal_condition, type, terminal_condition: token > $o).
tff(runaway_loop, type, runaway_loop: token > $o).

tff(tok_eos, type, tok_eos: token).
tff(tok_im_end, type, tok_im_end: token).
tff(tok_space, type, tok_space: token).
tff(tok_one, type, tok_one: token).

tff(ax_eos_def, axiom, is_eos(tok_eos) & is_eos(tok_im_end)).
tff(ax_rep_def, axiom, is_whitespace_rep(tok_space) & is_whitespace_rep(tok_one)).
tff(ax_distinct, axiom, (~ is_eos(tok_space)) & (~ is_eos(tok_one))).

tff(ax_term, axiom, ! [T: token] : (is_eos(T) => (terminal_condition(T) & (~ runaway_loop(T))))).
tff(ax_runaway, axiom, ! [T: token] : (is_whitespace_rep(T) => (~ terminal_condition(T)))).

tff(conjecture_separation, conjecture,
    ! [T: token] : (is_eos(T) => (~ is_whitespace_rep(T)))).
