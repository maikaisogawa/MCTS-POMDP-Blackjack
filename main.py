# pylint: disable = C
'''Main script'''

import sys

import pandas as pd
import numpy as np
import random
from scipy.special import gamma, factorial
import scipy
import collections
from datetime import datetime
from collections import defaultdict

### GENERAL NOTES - helful for write-up ###############################################################################
# RULES: - specific rules implemented for this problem
# reshuffle deck if less than 1/4 left? --> look up actual rule for this
# House must hit on 16
# win on 21, bust over 21

# RELAXATIONS: - what we restricted to make the problem easier to model
# suit of card does not matter in Blackjack
# bet amount is not variable turn by turn
# not allowing split hands (if two of same value dealt)

# OTHER: - other interesting things to mention in the write-up
# relatively small state space, so can represent belief probabilities
# interesting to compare immediate hand vs. counting card outcome
# compare to human performance best on 'best practices'
#######################################################################################################################

### QUESTIONS TO TEAM #################################################################################################
# how do we want to deal with the value of Ace? --> maybe use always as 11 unless causes us to bust right away
# we need to implement game play with human best-practices to compare our POMDP model to what humans would typically do
# figure out how to shuffle the deck - currently represented as an array, we can just 'pull from the top'
# if counting cards, state space also includes current contents of deck
# may take multiple actions during a turn (ex. hit, hit, stay) -- will we have to represent actions as histories?
# - realistically probably can't hit more that 3 times (we can set this limit)
# - OR just store the value of full hand in state in Q and whether to hit or not - might not work for counting
#######################################################################################################################


# input: number of decks to use in game
# output: an array of full play deck
def buildDeck(NUM_DECKS):
    deck = []
    values = ["ACE", "KING", "QUEEN", "JACK", "TEN", "NINE", "EIGHT", "SEVEN", "SIX", "FIVE", "FOUR", "THREE", "TWO", "ONE"]
    for value in values:
        for i in range(NUM_DECKS * 4):
            deck.append(value)
    return deck

# output: dictionary where key = "TEN", value = 10
def buildValueDictionary():
    valueDictionary = {} # key = "TEN", value = 10
    valueDictionary["ACE"] = 11
    valueDictionary["KING"] = 10
    valueDictionary["QUEEN"] = 10
    valueDictionary["JACK"] = 10
    valueDictionary["TEN"] = 10
    valueDictionary["NINE"] = 9
    valueDictionary["EIGHT"] = 8
    valueDictionary["SEVEN"] = 7
    valueDictionary["SIX"] = 6
    valueDictionary["FIVE"] = 5
    valueDictionary["FOUR"] = 4
    valueDictionary["THREE"] = 3
    valueDictionary["TWO"] = 2
    valueDictionary["ONE"] = 1
    return valueDictionary

# Train the agent on BlackJack and let them count cards! Remember the history of the deck and make informed decisions
def computeWithCount():
    # Transition probabilities probably really come into effect here
    pass

# Train the agent on BlackJack without letting them count cards - no histories - immediate hand decisions only
def computeWithoutCount():
    NUM_DECKS = 2 # constant to change how many decks we are playing with
    AMOUNT_BET = 5
    deck = buildDeck(NUM_DECKS)
    valueDictionary = buildValueDictionary()
    maxCards = len(deck)
    # shuffle deck? --> function to shuffle elements in an array?
    
    # Q(s, a)
    # s : your hand total, card showing on dealer's hand, belief of what card the dealer has not showing
    # a : hit or stay
    # T(s' | s, a) = N(s, a, s') / N(s, a)
    # R : double bet amount (bet amount is not variable) if win, lose money if bust or lose
#        Q = np.zeros((numPossibleStates, 9)) # initialize Q - 312020 possible states, 9 possible actions
#        discount = 0.95
#        loop = 100 # arbitrary range
#
#        for i in range(loop):
#            for j in range(len(dataframe)): # for each line
#                s, a, r, sp = dataframe[j] # grab the variables
#                Q[s - 1][a - 1] += 0.01 * (r + (discount * np.max(Q[sp - 1]) - Q[s - 1][a - 1])) # Q learning update
#            print(i)
#
#        policy = np.zeros(numPossibleStates, dtype = "int") # initialize best policy array
#        policy = policy + 1
#
#        for i in range(numPossibleStates):
#            policy[i] = np.argmax(Q[i]) + 1

    # training portion
    # NUM_TRAINING_ROUNDS = 10 # constant to determine how many hands we want to train
    # #Q = np.zeros((21, 11), 2)) # initialize Q(s, d, a) - 22*10 possible states (hand max is 22 (two aces), observable dealer hand max is 11), 2 possible actions (hit or stay)
    # mark = 0 # keep track of where in the deck we are
    # for i in range(NUM_TRAINING_ROUNDS):
    #     if float(maxCards - mark) / float(maxCards) < 0.25: # reshuffle deck if not enough cards left in it
    #         mark = 0
    #         # shuffle deck
    #     # deal hands:
    #     agentHandValue = 0
    #     agentHand = []
    #     for j in range(2): # deal two cards to agent
    #         agentHand.append(deck[mark])
    #         agentHandValue += valueDictionary[deck[mark]]
    #         mark += 1
    #     if agentHandValue > 21: # this is because they have two Aces
    #         agentHandValue -= 10 # choose 1 as value of Ace instead of 11
    #     dealerHandValue = 0
    #     dealerHand = []
    #     for j in range(2): # deal two cards to dealer - first one hidden from observable hand value
    #         dealerHand.append(deck[mark])
    #         mark += 1
    #     dealerHandValue += valueDictionary[dealerHand[1]] # only observe second card

    #     # player action - exploratory
    #     bust = 0
    #     win = 0
    #     while (bust != 1) and (win != 1):
    #         if agentHandValue == 21:
    #             win = 1
    #             reward = AMOUNT_BET
    #             # update Q
    #             continue
    #         action = random.choice(1, 2) # hit is 1, stay is 2
    #         if action == 1: # hit
    #             state = agentHandValue
    #             newCard = deck[mark]
    #             mark += 1
    #             newCardValue = valueDictionary[newCard]
    #             if agentHandValue + newCardValue > 21:
    #                 bust = 1
    #                 reward = -(AMOUNT_BET)
    #             elif agentHandValue + newCardValue == 21:
    #                 win = 1
    #                 reward = AMOUNT_BET
    #             #elif agentHandValue + newCardValue < 21:
                    
    #                 # keep playing
    #             # do something
    #         elif action == 2: # stay
    #             # do something else
    #     # if bust, take money and move on
    #     # dealer action based on rules - if bust, automatic win
    #     # compare hands
    #     # distribute reward - update Q
        
    # # get best policy

    # # playing portion - play by applying best policy
    # NUM_PLAYING_ROUNDS = 10 # constant to determine how many hands we want to play
    # # train iteself vs. us inputting human best-practices? (stay on 17)
    # # see how much money we win over a certain # of rounds
    pass

#
def main():
    startTime = datetime.now() # start timer
    if len(sys.argv) != 1:
        raise Exception("usage: python main.py")
#    count = sys.argv[1]
#    if count == "withCount":
#        computeWithCount()
#    elif count == "withoutCount":
#        computeWithoutCount()
    computeWithoutCount()
    print(datetime.now() - startTime) # print the runtime for README.pdf

if __name__ == '__main__':
    main()
