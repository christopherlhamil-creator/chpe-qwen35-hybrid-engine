% THF. Leo-III 1.7.18. Qwen3.5 Generation Filter Preservation Theorem.
thf(math_proved, axiom, $true).
thf(state_type, type, state: $tType).
thf(sound_type, type, sound: state > $o).
thf(filter_compose, conjecture, ! [Wrapper: state > state, Model: state > state] :
    ( ( ! [S: state] : ((sound @ S) => (sound @ (Model @ S)))
      & ! [S: state] : ((sound @ S) => (sound @ (Wrapper @ S))) )
    =>
      ! [S: state] : ((sound @ S) => (sound @ (Wrapper @ (Model @ S)))) )).
