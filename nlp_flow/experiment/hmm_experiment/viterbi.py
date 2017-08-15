# encoding=utf8

"""
https://en.wikipedia.org/wiki/Viterbi_algorithm
"""

NORMAL = 'normal'
COLD = 'cold'
DIZZY = 'dizzy'
HEALTHY = 'healthy'
FEVER = 'fever'

obs = (NORMAL, COLD, DIZZY)
states = (HEALTHY, FEVER)
start_p = {HEALTHY: 0.6, FEVER: 0.4}
trans_p = {
    HEALTHY: {HEALTHY: 0.7, FEVER: 0.3},
    FEVER: {HEALTHY: 0.4, FEVER: 0.6}
}
emit_p = {
    HEALTHY: {NORMAL: 0.5, COLD: 0.4, DIZZY: 0.1},
    FEVER: {NORMAL: 0.1, COLD: 0.3, DIZZY: 0.6}
}


def viterbi(obs):
    prev_states = []
    # initial
    start_state = {}
    for state in states:
        start_state[state] = ('start', start_p[state] * emit_p[state][obs[0]])
    prev_states.append(start_state)
    """
    prev_states = [
        {
            HEALTHY:('prev state','prob'),
            FEVER:('prev state', 'prob')
        }
    ]
    """
    # mid
    for ob in obs[1:]:
        curr_states = {}
        for curr in states:
            # all prev states to the curr state
            prev_path = [(prev, prev_states[-1][prev][1] * trans_p[prev][curr] * emit_p[curr][ob]) for prev in states]
            # select max prob path
            curr_states[curr] = max(prev_path, key=lambda item: item[1])
        prev_states.append(curr_states)
    # backtracking the longest path
    ret = []
    end_state = max([(state, prev_states[-1][state][1]) for state in states], key=lambda item: item[1])[0]
    ret.append(end_state)
    last_state = end_state
    for i in range(len(prev_states) - 1, 0, -1):
        last_state = prev_states[i][last_state][0]
        ret.append(last_state)
    return [i for i in reversed(ret)]


# from ipdb import set_trace as st
# st(context=21)
print viterbi(obs)
