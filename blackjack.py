# pylint: disable = C, no-else-return, no-else-break
# pylint: disable = too-many-instance-attributes, too-few-public-methods, invalid-sequence-index
import itertools
import numpy as np
import collections
import operator as op
from functools import reduce
import copy

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

def action_space():
    return [0, 1]

def state_space(max_card_val):
    return list(itertools.product([0, 1, 2, 3, 4], repeat=max_card_val))

#state of 1 dealer and 1 player = (cards in deck, cards in player's hand, cards in dealer's hand)

def calc_handscore(hand):
    score = 0
    for card in range(1,len(hand)):
        score += (card+1)*hand[card]
    if hand[0] == 0:
        #no aces in hand
        return score
    else:
        #hand[0] special case for aces
        ace_value = list(itertools.product([1,11],repeat=hand[0]))
        possible_scores = []
        for entry in ace_value:
            possible_comb_ace = list(entry)
            possible_comb_ace.append(score)
            local_score = sum(possible_comb_ace)
            possible_scores.append(local_score)
        possible_scores.sort(reverse=True)
        for entry in possible_scores:
            if entry <= 21:
                return entry
        return possible_scores[len(possible_scores)-1] #bust!

def naive_policy0(state):
    if calc_handscore(state[1]) < 16:
        return 1
    else:
        return 0

def naive_policy1(state):
    if calc_handscore(state[1]) < 12:
        return 1
    else:
        return 0

def random_policy(state):
    return np.random.randint(0,2)

def belief_policy(state):
    pseudo_dealer = state[2].copy()
    if calc_handscore(state[2]) < 17:
        drawn_card = draw_card_from_deck(state[0])
        pseudo_dealer[drawn_card] += 1
    if calc_handscore(pseudo_dealer) > 21:
        return 0
    if calc_handscore(state[1]) > calc_handscore(pseudo_dealer):
        return 0
    else:
        return 1

def draw_card_from_deck(deck):
    possible_cards = []
    for card in range(0, len(deck)):
        if deck[card] > 0:
            for _ in range(0, deck[card]):
                possible_cards.append(card)
    if len(possible_cards) == 0:
        return -1
    else:
        drawn_card = possible_cards[np.random.randint(0, len(possible_cards))] 
        return drawn_card

def compare_hand(player,dealer):
    playscore = calc_handscore(player)
    dealscore = calc_handscore(dealer)
    if playscore > 21:
        return -1
    if dealscore > 21:
        return 1
    if playscore > dealscore:
        return 1
    elif playscore == dealscore:
        return 0
    else:
        return -1

def generate_new_game(state):
    #print("generating new game!")
    new_state = copy.deepcopy(state)
    new_state[1] = [0 for i in range(0,len(state[1]))]
    new_state[2] = [0 for i in range(0,len(state[2]))]
    for player in range(0,(len(state)-1)*2):
        drawn_card = draw_card_from_deck(new_state[0])
        if drawn_card == -1:
            #no more cards in deck
            return end_state()
        new_state[0][drawn_card] += -1
        new_state[player%2+1][drawn_card] += 1
    return new_state

#TODO pylint warns of too many branches
def generate_nstate_reward(state,action,verbose=False): 
    if action == 1:
        drawn_card = draw_card_from_deck(state[0])
        if drawn_card == -1:
            #no more cards in deck
            return (end_state(),compare_hand(state[1],state[2]))
        new_state = copy.deepcopy(state)
        new_state[0][drawn_card] += -1
        new_state[1][drawn_card] += 1
        handscore = calc_handscore(new_state[1])
        if handscore > 21:
            #bust! generate another game
            if verbose:
                print("agent chose to hit and busted!",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
            new_state = generate_new_game(new_state)
            return (new_state,-1)
        elif handscore == 21:
            #blackjack!
            if verbose:
                print("agent chose to hit and got blackjack!",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
            new_state = generate_new_game(new_state)
            return (new_state,1)
        else:
            #print("agent chose to hit! agent=",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
            return (new_state,0)
    else:
        #action == 0
        new_state = copy.deepcopy(state)
        while calc_handscore(new_state[2]) < 17:
            drawn_card = draw_card_from_deck(new_state[0])
            new_state[0][drawn_card] += -1
            new_state[2][drawn_card] += 1
        reward = compare_hand(new_state[1],new_state[2])
        if verbose:
            if reward == 1:
                print("agent chose to stay and won!",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
            elif reward == -1:
                print("agent chose to stay and lost!",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
            else:
                print("agent chose to stay and tied!",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))

        #generate new game
        new_state = generate_new_game(new_state)
        return (new_state,reward)

def rollout(gamma,state,depth,policy_func):
    if depth == 0 or state == end_state():
        return 0
    action = policy_func(state)
    next_state,reward = generate_nstate_reward(state,action)
    return reward + gamma*rollout(gamma,next_state,depth-1,policy_func)

def init_rollout(gamma,state,depth,policy_func,init_action):
    if depth == 0 or state == end_state():
        return 0
    next_state,reward = generate_nstate_reward(state,init_action)
    return reward + gamma*rollout(gamma,next_state,depth-1,policy_func)


def initial_state():
    deck = [4 for i in range(0,10)]
    hand = [0 for i in range(0,10)]
    return [deck,hand.copy(),hand.copy()]

def end_state():
    deck = [0 for i in range(0,10)]
    hand = [0 for i in range(0,10)]
    return [deck,hand.copy(),hand.copy()]

class MCTS:
    def __init__(self):
        self.h = {}
        self.T = set()
        self.N_h = collections.defaultdict(int)
        self.N_ha = collections.defaultdict(int)
        self.Q = collections.defaultdict(float)
        self.N_ha_nought = {}
        self.Q_nought = {}
        self.c = 1.
        self.gamma = 1.

def generate_new_game_withobserv(state):
    #print("generating new game!")
    observation = []
    new_state = state
    state[1] = [0 for i in range(0,len(state[1]))]
    state[2] = [0 for i in range(0,len(state[2]))]
    for player in range(0,(len(state)-1)*2):
        drawn_card = draw_card_from_deck(new_state[0])
        observation.append(drawn_card)
        if drawn_card == -1:
            #no more cards in deck
            return (end_state(),observation)
        new_state[0][drawn_card] += -1
        new_state[player%2+1][drawn_card] += 1
    return (new_state,observation)

#TODO pylint warns of too many branches
#(s',o,r) ~ G(s,a)
def generate_nstate_observ_reward(state,action,verbose=False):
    observation = []
    if action == 1:
        drawn_card = draw_card_from_deck(state[0])
        observation.append(drawn_card)
        if drawn_card == -1:
            #no more cards in deck
            return (end_state(),tuple(observation),compare_hand(state[1],state[2]))
        new_state = copy.deepcopy(state)
        new_state[0][drawn_card] += -1
        new_state[1][drawn_card] += 1
        handscore = calc_handscore(new_state[1])
        if handscore > 21:
            #bust! generate another game
            if verbose:
                print("agent chose to hit and busted!",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
            new_state,newobserv = generate_new_game_withobserv(new_state)
            observation.extend(newobserv)
            return (new_state,tuple(observation),-1)
        elif handscore == 21:
            #blackjack!
            if verbose:
                print("agent chose to hit and got blackjack!",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
            new_state,newobserv = generate_new_game_withobserv(new_state)
            observation.extend(newobserv)
            return (new_state,tuple(observation),1)
        else:
            if verbose:
                print("agent chose to hit! agent=",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
            return (new_state,tuple(observation),0)
    else:
        #action == 0
        new_state = copy.deepcopy(state)
        while calc_handscore(new_state[2]) < 17:
            drawn_card = draw_card_from_deck(new_state[0])
            observation.append(drawn_card)
            new_state[0][drawn_card] += -1
            new_state[2][drawn_card] += 1
        reward = compare_hand(new_state[1],new_state[2])
        if verbose:
            if reward == 1:
                print("agent chose to stay and won!",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
            elif reward == -1:
                print("agent chose to stay and lost!",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
            else:
                print("agent chose to stay and tied!",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))

        #generate new game
        new_state,newobserv = generate_new_game_withobserv(new_state)
        observation.extend(newobserv)
        return (new_state,tuple(observation),reward)

def simulate(mcts,state,history,depth):
    if depth == 0 or state == end_state():
        return 0
    if history not in mcts.T:
        for act in action_space():
            if (history,act) in mcts.N_ha_nought:
                mcts.N_ha[(history,act)] = mcts.N_ha_nought[(history,act)]
            if (history,act) in mcts.Q_nought:
                mcts.Q[(history,act)] = mcts.Q_nought[(history,act)]
            if act == 0:
                mcts.Q[(history,act)] = mcts.Q_nought[(history,act)] = (-1. + calc_handscore(state[1])/11.)
        mcts.T.add(history)
        return rollout(mcts.gamma,state,depth,belief_policy)

    #argmax_a Q(h,a) + csqrt(logN(h)/N(h,a))
    opt_act = 0
    max_q = float('-inf')
    tmp_qval = []
    mcts.N_h[history] = sum([mcts.N_ha[history,act] for act in action_space()])
    for act in action_space():
        if mcts.N_ha[(history,act)] > 0 and mcts.N_h[history] > 0:
            q_val = mcts.Q[(history,act)] + mcts.c*np.sqrt(np.log(mcts.N_h[history])/mcts.N_ha[(history,act)])
        else:
            q_val = mcts.Q[(history,act)]
        if q_val > max_q:
            max_q = q_val
            opt_act = act
        tmp_qval.append(q_val)
    if tmp_qval[0] == tmp_qval[1]:
        #no optimal action, choose random action
        opt_act = np.random.randint(0,2)

    #(s',o,r) ~ G(s,a)
    new_state,observation,reward = generate_nstate_observ_reward(state,opt_act)
    newhistory = copy.deepcopy(history)
    newhistory = newhistory + (opt_act,observation)
    new_q = reward + mcts.gamma* simulate(mcts,new_state,newhistory,depth-1)
    mcts.N_ha[(history,opt_act)] += 1
    mcts.Q[(history,opt_act)] += (new_q - mcts.Q[(history,opt_act)])/mcts.N_ha[(history,opt_act)]
    #print("updating Q(",history,opt_act,") to ",mcts.Q[(history,opt_act)] )
    return new_q

def get_all_possible_fulldeckstarts():
    states = []
    hidden_cards = []
    agenthand = itertools.combinations(range(0,10), 2)
    dealerhand = itertools.permutations(range(0,10), 2)
    for agent in agenthand:
        for dealer in dealerhand:
            s = initial_state()
            for card in agent:
                s[0][card] += -1
                s[1][card] += 1
            for card in dealer:
                s[0][card] += -1
                s[2][card] += 1
            hidden_cards.append(dealer[1])
            states.append(s)
    return (states,hidden_cards)

def possiblebelief(belief_state):
    init_deck = belief_state[0]
    agenthand = belief_state[1]
    observed_dealerhand = belief_state[2]
    possible_states = []
    for card in range(0,len(init_deck)):
        if init_deck[card] > 0:
            possible_deck = init_deck.copy()
            possible_deck[card] += -1
            possible_dealerhand = observed_dealerhand.copy()
            possible_dealerhand[card] += 1
            for _ in range(0,init_deck[card]):
                possible_states.append([possible_deck,agenthand.copy(),possible_dealerhand])
    return possible_states


def selectaction(mcts,beliefstate,depth,max_iters):
    history = tuple()
    all_possible_states = possiblebelief(beliefstate)
    for _ in range(0, max_iters):
        #print("sampling belief!")
        sampled_state = copy.deepcopy(all_possible_states[np.random.randint(0,len(all_possible_states))])
        #print("sampled: agent=",calc_handscore(sampled_state[1])," dealer=",calc_handscore(sampled_state[2]))
        simulate(mcts,sampled_state,history,depth)

    max_q = float('-inf')
    opt_act = 0
    tmp_qval = []
    for act in action_space():
        if mcts.Q[(history,act)] > max_q:
            max_q = mcts.Q[(history,act)]
            opt_act = act
        tmp_qval.append(mcts.Q[(history,act)])
    if tmp_qval[0] == tmp_qval[1]:
        #no optimal action, choose random action
        opt_act = np.random.randint(0,2)
    #print("value for stay=",tmp_qval[0]," value for hit=",tmp_qval[1])
    return opt_act

def update_beliefstate(beliefstate,state,reward):
    if reward != 0:
        if state == end_state():
            return None
        #game finished, pick new hidden card
        new_beliefstate = copy.deepcopy(state)
        dealer_cards = []
        for card in range(0,len(state[2])):
            for _ in range(0,state[2][card]):
                dealer_cards.append(card)
        hidden_card = dealer_cards[np.random.randint(0,2)]
        new_beliefstate[2][hidden_card] += -1
        new_beliefstate[0][hidden_card] += 1
        return new_beliefstate
    else:
        #game continues, meaning agent hit without busting
        new_beliefstate = copy.deepcopy(beliefstate)
        new_beliefstate[1] = state[1].copy()
        return new_beliefstate
        #agent has vision of new card in his hand

#TODO pylint warns of too many branches and too many local variables
def game_simulate(n_games,shuffle_freq,policy=None,verbose=False):
    game_count = 0
    total_utility = 0.
    OutOfCards = False
    while game_count < n_games:
        if game_count%shuffle_freq == 0 or OutOfCards is True:
            #print("re-shuffling deck!")
            OutOfCards = False
            init_s = generate_new_game(initial_state())
            dealer_cards = []
            for card in range(0,len(init_s[2])):
                for _ in range(0,init_s[2][card]):
                    dealer_cards.append(card)

            #create belief_state for MCTS sampling
            belief_state = copy.deepcopy(init_s)
            hidden_card = dealer_cards[np.random.randint(0,2)]
            belief_state[2][hidden_card] += -1
            belief_state[0][hidden_card] += 1
            next_state = copy.deepcopy(init_s)

        game_count += 1
        while True:
            if policy is None:
                mcts = MCTS()
                mcts.c = 5
                mcts.gamma = 0.5
                action = selectaction(mcts,belief_state,10,100)
            else:
                action = policy(belief_state)
            if verbose:
                if action == 1:
                    print("agent chose to hit. agent=",calc_handscore(belief_state[1]),belief_state[1]," observed_dealer=",calc_handscore(belief_state[2]),"real_dealer=",calc_handscore(next_state[2]))
                else:
                    print("agent chose to stay. agent=",calc_handscore(belief_state[1])," dealer=",calc_handscore(belief_state[2]),"real_dealer=",calc_handscore(next_state[2]))
            next_state,reward = generate_nstate_reward(next_state,action)
            belief_state = update_beliefstate(belief_state,next_state,reward)
            if next_state == end_state():
                OutOfCards = True
                if verbose:
                    print("deck ran out of cards!")
                break
            total_utility += reward
            if reward == 1 :
                if verbose:
                    print("agent wins!")
                break
            elif reward == -1:
                if verbose:
                    print("agent lost!")
                break
            elif action == 0 and reward == 0:
                if verbose:
                    print("agent tied!")
                break
    return total_utility

if __name__ == '__main__':
    np.random.seed(0)
    print("MCTS=",game_simulate(50,10))
    print("naive0=",game_simulate(50,10,naive_policy0))
    print("naive1=",game_simulate(50,10,naive_policy1))
    print("random=",game_simulate(50,10,random_policy))

"""
#init_s = generate_new_game(initial_state())
init_s = [[3, 4, 4, 4, 3, 4, 3, 4, 3, 4], [1, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 1, 0]]
dealer_cards = []
for card in range(0,len(init_s[2])):
    for i in range(0,init_s[2][card]):
        dealer_cards.append(card)

belief_state = copy.deepcopy(init_s)
hidden_card = dealer_cards[np.random.randint(0,2)]
belief_state[2][hidden_card] += -1
belief_state[0][hidden_card] += 1
#all_possible_states = possiblebelief(belief_state)
#sampled_state = copy.deepcopy(all_possible_states[np.random.randint(0,len(all_possible_states))])
#print(sampled_state)
#print(init_s)
#print("sampled: agent=",calc_handscore(sampled_state[1])," dealer=",calc_handscore(sampled_state[2]))
#print(init_s)
#print("true: agent=",calc_handscore(init_s[1])," dealer=",calc_handscore(init_s[2]))
#print("visible dealer=",calc_handscore(belief_state[2]))
"""

