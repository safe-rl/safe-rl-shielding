import math

# generate dfa file
MAX_DEPTH = 95
MAX_OXYGEN = 105

NUM_BITS_DEPTH = int(math.ceil(math.log(MAX_DEPTH + 1, 2)))
K_OFFSET = NUM_BITS_DEPTH + 1
OXYGEN_LOW = NUM_BITS_DEPTH + 3
OXYGEN_FULL = NUM_BITS_DEPTH + 4
DIVER_FOUND = NUM_BITS_DEPTH + 5
ACTION_OFFSET = NUM_BITS_DEPTH + 6

with open("seaquest.dfa", "w") as file:

    transitions = []
    bad_state = MAX_OXYGEN + 1
    # for all good states
    for state in range(1, MAX_OXYGEN + 1):
        # for all good actions
        for action in range(9):
            action_enc = [str(-(idx + ACTION_OFFSET) if x == '0' else (idx + ACTION_OFFSET)) for idx, x in enumerate(list(bin(action)[2:].rjust(4, '0')))]
       
            # go to FULL if OXYGEN_FULL flag is 1, all actions allowed
            transitions.append("{0} {1} {2} {3}".format(state, MAX_OXYGEN, OXYGEN_FULL, " ".join(action_enc)))
            
            # for all passed frames
            for k in range(1,5):
                k_enc = [str(-(idx + K_OFFSET) if x == '0' else (idx + K_OFFSET)) for idx, x in enumerate(list(bin(k - 1)[2:].rjust(2, '0')))]

                # not allowed to go to the surface if OXYGEN_LOW = 0 and DIVER_FOUND = 0
                if state == MAX_OXYGEN:
                    transitions.append("{0} {0} {1} -{2} -{3} {4} {5}".format(state, " ".join(k_enc), OXYGEN_LOW, OXYGEN_FULL, DIVER_FOUND, " ".join(action_enc)))
                    for depth in range(2 ** NUM_BITS_DEPTH):
                        depth_enc = [str(-(idx + 1) if x == '0' else (idx + 1)) for idx, x in enumerate(list(bin(depth)[2:].rjust(NUM_BITS_DEPTH, '0')))]

                        allowed_actions = [0, 2, 3, 4, 7, 8] if depth <= 4 else range(9)
                        target_state = MAX_OXYGEN if (action in allowed_actions) else bad_state
                        transitions.append("{0} {1} {2} {4} -{5} -{6} -{7} {3}".format(state, target_state, " ".join(depth_enc), " ".join(action_enc), " ".join(k_enc), OXYGEN_LOW, OXYGEN_FULL, DIVER_FOUND))
                                    
                for depth in range(2 ** NUM_BITS_DEPTH):
                    depth_enc = [str(-(idx + 1) if x == '0' else (idx + 1)) for idx, x in enumerate(list(bin(depth)[2:].rjust(NUM_BITS_DEPTH, '0')))]
                    if state - k >= depth:
                        allowed_actions = range(9)
                    else:
                        allowed_actions = [1, 5, 6]

                    target_state = max(1, state - k) if action in allowed_actions else bad_state
                    if state != MAX_OXYGEN:
                        transitions.append("{0} {1} {2} {4} -{5} {3}".format(state, target_state, " ".join(depth_enc), " ".join(action_enc), " ".join(k_enc), OXYGEN_FULL))
                    else:
                        transitions.append("{0} {1} {2} {4} {5} -{6} {3}".format(state, target_state, " ".join(depth_enc), " ".join(action_enc), " ".join(k_enc), OXYGEN_LOW, OXYGEN_FULL))

        for action in range(9, 16):
            action_enc = [str(-(idx + ACTION_OFFSET) if x == '0' else (idx + ACTION_OFFSET)) for idx, x in enumerate(list(bin(action)[2:].rjust(4, '0')))]
            transitions.append("{0} {1} {2}".format(state, bad_state, " ".join(action_enc)))
    
    transitions.append("{0} {0}".format(bad_state))
    file.write("dfa {0} {3} 4 1 1 {1}\n{2}\n{0}\n".format(bad_state, len(transitions), MAX_OXYGEN, NUM_BITS_DEPTH + 5))
    file.write("\n".join(transitions) + "\n")
    for bit in range(1, NUM_BITS_DEPTH + 1):
        file.write("{0} depth{1}\n".format(bit, NUM_BITS_DEPTH - bit + 1))
    file.write("{0} k2\n".format(K_OFFSET))
    file.write("{0} k1\n".format(K_OFFSET + 1))
    file.write("{0} oxygen_low\n".format(OXYGEN_LOW))
    file.write("{0} oxygen_full\n".format(OXYGEN_FULL))
    file.write("{0} diver_found\n".format(DIVER_FOUND))
    
    for bit in range(1, 5):
        file.write("{0} action{1}\n".format(ACTION_OFFSET - 1 + bit, 5 - bit))

