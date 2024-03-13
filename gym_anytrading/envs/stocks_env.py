import numpy as np

from .trading_env import TradingEnv, Actions, Positions


# Inherits the TradingEnv. TradingEnv is like an abstract class where the reward
# calculations are not included. It just sets up the framework for the stockenv and forexenv

'''
env.reset and env.step functions are inherited. Note that automatically position is set to short.
the short position in this package really just means we're not holding any shares since 
shorting mechanisms are not implemented.
'''
class StocksEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, render_mode=None):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size, render_mode)

        self.trade_fee_bid_percent = 0.01  # unit (buying fees)
        self.trade_fee_ask_percent = 0.005  # unit (selling fees)

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()  # use close price as current share price

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)  # price difference added to features for model training
        signal_features = np.column_stack((prices, diff))  # we will include indicators by overriding this function.
        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _calculate_reward(self, action):
        ''' This reward function only considers long positions really, even though it has the option for
        short position. Long position reward calculated as expected. Reward isn't applied to long position until a sell action'''
        step_reward = 0

        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Long:
                step_reward += price_diff

        return step_reward

    def _update_profit(self, action):
        '''Similar to the reward calculation, this function doesn't really calculate profits from short positions;
        only long positions.'''
        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        if trade or self._truncated:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                print("SOLD!!"+"-"*150)
                print("shares:", shares)
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price
                print("total profit", self._total_profit)


    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit
