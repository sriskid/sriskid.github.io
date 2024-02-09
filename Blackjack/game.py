import numpy as np
import random

class BlackJack:
    def __init__(self, num_decks:int):
        # Initialize the game elements
        self.reset(num_decks)

    def shuffle_deck(self):
        random.shuffle(self.deck)

    def reset(self, num_decks:int):
        """
        Initializes the game by creating a new deck, giving the player two cards and the dealer one card (face up).
        """
        # Initialize the game again
        self.deck = np.repeat([2, 3, 4, 5, 6, 7, 8, 9, 10, "J", "Q", "K", "A"], 4*num_decks).tolist()
        self.shuffle_deck()
        self.player_hand = [self.draw_card(), self.draw_card()]
        self.dealer_hand = [self.draw_card()]
        self.actions = ["Hit", "Stand"]
        self.episode_ended = False

    def step(self, action:str):
        """
        Plays one round of the game. The episode ends when the entire game finishes.
        """
        if self.episode_ended:
            raise ValueError("The episode has already ended. Please reset the game.")
        reward = 0

        # Plays one round of the game
        # Player chooses to hit or stand
        if action == "Hit":
            self.player_hand.append(self.draw_card())
            if self.calculate_hand_total(self.player_hand) > 21:
                self.episode_ended = True
                return self.get_state(), -1, True # You Bust, Episode Ends
        
        # if not self.episode_ended:
        #     self.bot_turn()
        elif action == "Stand":
            self.episode_ended = True

        # The dealer plays
        if self.episode_ended:
            player_hand_value = self.calculate_hand_total(self.player_hand)
            self.dealer_hand.append(self.draw_card())
            while self.calculate_hand_total(self.dealer_hand) < 17:
                self.dealer_hand.append(self.draw_card())
                if self.check_game_over():
                    break
            reward = self.calculate_reward()
            return self.get_state(), reward, True

        return self.get_state(), 0, False # Game is still in progress

    def get_state(self):
        """
        Gets the current state of the game, which is the player's hand value and the dealer's upcard
        """
        dealer_upcard = self.card_value(self.dealer_hand[0])
        player_hand_value = self.calculate_hand_total(self.player_hand)

        state = (player_hand_value, dealer_upcard)
        return state

    def draw_card(self):
        card = self.deck.pop()
        return card

    def check_game_over(self):
        """
        Checks to see if the game is over based on the card value sums.
        """
        player_sum = self.calculate_hand_total(self.player_hand)
        dealer_sum = self.calculate_hand_total(self.dealer_hand)

        if player_sum > 21 or dealer_sum > 21:
            return True
        return False

    def card_value(self, card: str | int):
        """
        Calculates the value of a card whether it is a number card or a royal
        """
        if card in ["J", "Q", "K"]:
            return 10
        elif card == "A":
            return 11
        else:
            return int(card)

    def calculate_hand_total(self, hand:list[str | int]):
        """
        Calculates the total of the hand, including adjusting for aces
        """
        total = 0
        num_aces = 0
        for card in hand:
            if card in ["J", "Q", "K"]:
                total += 10
            elif card == "A":
                num_aces += 1
                total += 11
            else:
                total += int(card)
        
        # Adjust for aces if necessary
        if total > 21:
            for _ in range(num_aces):
                total -= 10
                if total < 21:
                    break
            
        return total

    def calculate_reward(self):
        """
        Calculates the reward based on the events of the game.
        """
        player_sum = self.calculate_hand_total(self.player_hand)
        dealer_sum = self.calculate_hand_total(self.dealer_hand)

        if player_sum > 21:
            return -1 # You bust, dealer wins

        elif dealer_sum > 21:
            return 2 # Dealer bust, you win

        elif player_sum < dealer_sum:
            return -1 # You had less than the dealer, dealer wins
        
        elif player_sum > dealer_sum:
            return 2 # You had more than the dealer, you win

        else:
            return 0 # You tied, push
    
