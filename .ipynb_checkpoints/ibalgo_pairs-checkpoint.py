import collections as col
import numpy as np
import queue
from ibapi import wrapper
from ibapi import client
from ibapi import contract
from ibapi import order
from collections import defaultdict
import os.path as opath
import datetime
import copy
import numpy as np
import datetime
from pytz import timezone
import pandas as pd


def get_third_friday(year, month):
    for day in range(15, 22):
        day = datetime.datetime(year, month, day)
        if day.weekday() == 4:
            return day

def date_to_contract(contract_date):
    third_friday = get_third_friday(contract_date.year, contract_date.month)
    if contract_date.month % 3 == 0:
        if contract_date > third_friday.date():
            if contract_date.month == 12:
                year = contract_date.year + 1
                month = 3
            else:
                year = contract_date.year
                month = contract_date.month + 3
        else:
            year = contract_date.year
            month = contract_date.month
    else:
        year = contract_date.year
        month = (int(contract_date.month / 3) + 1) * 3
    return year, month

# later enhance system to take long and short positions
class Entry(object):

    def __init__(self, entry_signal, stop_loss=1.00, profit_taker=1.00,
                 entry_price_offset=-0.25, trade_wait_tm=4, quantity=1,
                 action='BUY', pr_per_point=5.0, **kwargs):
        self.entry_signal = entry_signal
        self.stop_loss = stop_loss
        self.profit_taker = profit_taker
        self.entry_price_offset = entry_price_offset
        self.trade_wait_tm = trade_wait_tm
        self.quantity = quantity
        self.action = action
        self.sign =  1 if self.action == 'BUY' else -1
        self.wins = 0
        self.losses = 0
        self.entry_tms = []
        self.exit_tms = []
        self.profits = []
        self.cum_profits = []
        self.cum_profit = 0
        self.pr_per_point = pr_per_point
        self.in_position = False
        self.kwargs = kwargs

    def enter(self, online_indicator):
        return self.entry_signal(online_indicator, **self.kwargs)

    def entry(self, entry_tm):
        self.in_position = True
        self.entry_tms.append(entry_tm)

    def exit(self, exit_tm, win_loss):
        self.in_position = False
        self.exit_tms.append(exit_tm)
        self.profits.append(win_loss * self.pr_per_point)
        if win_loss == 1:
            self.wins +=1
        else:
            self.losses += 1
        if len(self.cum_profits) == 0:
            self.cum_profits.append(self.profits[-1])
        else:
            self.cum_profits.append(self.cum_profits[-1] + self.profits[-1])
        self.cum_profit = self.cum_profits[-1]


def scalping_trend(online_indicator, rsi_threshold=70, pr_diff_ewa_thresh=0, adx_threshold=15):
    indicators =  online_indicator.temp_indicator
    condition = indicators.rsi < rsi_threshold
    condition = condition and indicators.pr_diff_ewa > pr_diff_ewa_thresh
    condition = condition and indicators.mean_pr > indicators.sma_pr
    condition = condition and indicators.adx > adx_threshold
    condition = condition and indicators.plus_dir_ind > indicators.min_dir_ind
    condition = condition and indicators.mean_pr > indicators.ewa_pr
    return condition

def rsi_long(online_indicator, rsi_threshold=30, pr_diff_ewa_thresh=0, adx_threshold=15):
    indicators =  online_indicator.temp_indicator
    condition = indicators.rsi < rsi_threshold and indicators.adx > 5 and indicators.mean_pr > indicators.sma_pr
    return condition

def rsi_short(online_indicator, rsi_threshold=70, pr_diff_ewa_thresh=0, adx_threshold=15):
    indicators =  online_indicator.temp_indicator
    condition = indicators.rsi > rsi_threshold and indicators.adx > 15 and indicators.mean_pr < indicators.sma_pr
    return condition

def volume_pulse_long(online_indicator, vol_roc_std_thresh=2.0):
    indicators = online_indicator.temp_indicator
    vol_roc_std_norm = indicators.volume_roc / (indicators.sma_volume_roc_diff_std + 1e-24)
    entry_price_diff_ewa = indicators.pr_diff_ewa
    high_diff = indicators.high - (indicators.prev_high if indicators.prev_high is not None else indicators.high)
    condition = vol_roc_std_norm > vol_roc_std_thresh and high_diff > 0 and indicators.volume_roc_last < indicators.volume_roc and indicators.mean_pr > indicators.sma_pr
    return condition

def volume_pulse_short(online_indicator, vol_roc_std_thresh=2.0):
    indicators = online_indicator.temp_indicator
    vol_roc_std_norm = indicators.volume_roc / (indicators.sma_volume_roc_diff_std + 1e-24)
    entry_price_diff_ewa = indicators.pr_diff_ewa
    low_diff = indicators.low - (indicators.prev_low if indicators.prev_low is not None else indicators.low)
    condition = vol_roc_std_norm > vol_roc_std_thresh and low_diff < 0 and indicators.volume_roc_last < indicators.volume_roc and indicators.mean_pr < indicators.sma_pr
    return condition

def volume_pulse_long_reverse(online_indicator, vol_roc_std_thresh=2.0):
    indicators = online_indicator.temp_indicator
    vol_roc_std_norm = indicators.volume_roc / (indicators.sma_volume_roc_diff_std + 1e-24)
    entry_price_diff_ewa = indicators.pr_diff_ewa
    high_diff = indicators.high - (indicators.prev_high if indicators.prev_high is not None else indicators.high)
    condition = vol_roc_std_norm > vol_roc_std_thresh and high_diff > 0 and indicators.volume_roc_last < indicators.volume_roc and indicators.mean_pr < indicators.sma_pr
    return condition

def volume_pulse_short_reverse(online_indicator, vol_roc_std_thresh=2.0):
    indicators = online_indicator.temp_indicator
    vol_roc_std_norm = indicators.volume_roc / (indicators.sma_volume_roc_diff_std + 1e-24)
    entry_price_diff_ewa = indicators.pr_diff_ewa
    low_diff = indicators.low - (indicators.prev_low if indicators.prev_low is not None else indicators.low)
    condition = vol_roc_std_norm > vol_roc_std_thresh and low_diff < 0 and indicators.volume_roc_last < indicators.volume_roc and indicators.mean_pr > indicators.sma_pr
    return condition

def sto_long(online_indicator, sto_threshold=25):
    indicators =  online_indicator.temp_indicator
    condition = indicators.sto < sto_threshold
    return condition

class ValueIndexDeque(object):

    def __init__(self):
        self.value_deque = col.deque()
        self.index_deque = col.deque()

    def popleft(self):
        self.value_deque.popleft()
        self.index_deque.popleft()

    def pop(self):
        self.value_deque.pop()
        self.index_deque.pop()

    def appendleft(self, i, val):
        self.value_deque.appendleft(val)
        self.index_deque.appendleft(i)

    def size(self):
        return len(self.value_deque)


class MinMaxSlidingWindow(object):

    def __init__(self, window_size=200, pos_comparison=None):
        self.min_val_ind_deque = ValueIndexDeque()
        self.max_val_ind_deque = ValueIndexDeque()
        self.window_size = window_size
        self.pos_comparison = pos_comparison
        self.max_val = -np.inf
        self.min_val = np.inf
        if self.pos_comparison is None:
            self.pos_comparison = lambda x, y: (x - y)

    def update(self, num, pos, deque, update_max=True):
        comparator = (lambda x, y: x >= y) if update_max else (lambda x, y: y >= x)
        pop_left = comparator(num, deque.value_deque[0]) if deque.size() > 0 else False
        while pop_left:
            deque.popleft()
            pop_left = comparator(num, deque.value_deque[0]) if deque.size() > 0 else False
        deque.appendleft(pos, num)
        check = self.pos_comparison(pos, deque.index_deque[-1]) >= self.window_size
        while check:
            deque.pop()
            check = self.pos_comparison(pos, deque.index_deque[-1]) >= self.window_size

    def update_min(self, num, pos):
        self.update(num, pos, self.min_val_ind_deque, update_max=False)
        self.min_val = self.min_val_ind_deque.value_deque[-1]

    def update_max(self, num, pos):
        self.update(num, pos, self.max_val_ind_deque, update_max=True)
        self.max_val = self.max_val_ind_deque.value_deque[-1]

    def update_min_max(self, max_val, min_val, pos):
        self.update_min(min_val, pos)
        self.update_max(max_val, pos)
    ## add update min_max that passes in min max


class IndicatorHolder(object):

    def __init__(self, sma_period=200, rsi_period=14, pr_ewa_period=4, pr_diff_ewa_period=5,
                 sto_period=180, adx_period=14, vol_fast_period=2, vol_slow_period=200,
                 vol_sma_period=200, sto_second_normalizer=60
                 ):
        self.sma_period = sma_period
        self.vol_sma_period = vol_sma_period
        self.vol_roc_queue = queue.Queue(maxsize=self.vol_sma_period)
        self.vol_roc_diff_queue = queue.Queue(maxsize=self.vol_sma_period)
        self.sma_queue = queue.Queue(maxsize=self.sma_period)
        self.rsi_weight = 1.0 / rsi_period
        self.pr_ewa_weight =  1.0 / pr_ewa_period
        self.pr_diff_ewa_weight =  1.0 / pr_diff_ewa_period
        self.sto_period = sto_period
        self.sto_min_max_sw = MinMaxSlidingWindow(window_size=self.sto_period,
                                                  pos_comparison=lambda x, y: (x - y).total_seconds() / sto_second_normalizer)
        self.adx_period = adx_period
        self.vol_fast_period = vol_fast_period
        self.vol_slow_period = vol_slow_period
        self.ewa_pr = None
        self.rsi = None
        self.sto = None
        self.up_ewa = 0
        self.down_ewa = 0
        self.sma_pr = None
        self.pr_diff = None
        self.pr_diff_ewa = None
        self.mean_pr =  None
        self.high = None
        self.low = None
        self.close = None
        self.first_tr_sum_cnt = 0
        self.true_range = None
        self.true_range_sum = 0
        self.true_range_sm = None
        self.plus_dir_mov = None
        self.plus_dir_mov_sum = 0
        self.plus_dir_mov_sm = None
        self.min_dir_mov = None
        self.min_dir_mov_sum = 0
        self.min_dir_mov_sm = None
        self.plus_dir_ind = None
        self.min_dir_ind = None
        self.first_dx_sum_cnt = 0
        self.dir_mov_ind = None
        self.dir_mov_ind_sum = 0
        self.atr = None
        self.adx = 0
        self.prev_close = None
        self.prev_high = None
        self.prev_low = None
        self.current_sec_cnt = 0
        self.current_sec_pr_total = 0
        self.current_time = None
        self.initialized = False
        self.sma_first_pr = 0
        self.sma_ttl_pr = 0
        self.volume = 0
        self.fast_volume_ewa = 0
        self.slow_volume_ewa = 0
        self.volume_roc = 0
        self.volume_roc_last = 0
        self.sma_volume_roc = 0
        self.sma_volume_roc_diff = 0
        self.sma_first_vol_roc = 0
        self.sma_first_vol_roc_diff = 0
        self.sma_ttl_vol_roc = 0
        self.sma_ttl_vol_roc_diff = 0
        self.sma_volume_roc_diff_std = 0
        self.eps = 1e-24
        self.open = 0
        self.sto_high = 0
        self.sto_low = 0

    def initialize(self, price, size, time):
        self.ewa_pr = price
        self.sma_pr = price
        self.mean_pr = price
        self.high = price
        self.low = price
        self.sto_high = price
        self.sto_low = price
        self.close = price
        self.open = price
        self.current_sec_pr_total = 0
        self.current_sec_cnt = 0
        self.rsi = 50
        self.sto = 50
        self.pr_diff = 0
        self.pr_diff_ewa = 0
        self.initialized = True
        self.current_time = time
        self.volume = size

    def update_price(self, price, size):
        self.current_sec_pr_total += price
        self.current_sec_cnt += 1.0
        self.mean_pr = self.current_sec_pr_total / self.current_sec_cnt
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += size

    def update_price_OHLC(self, bar):
        self.current_sec_pr_total += bar.close
        self.current_sec_cnt += 1.0
        self.mean_pr = bar.close
        self.high = bar.high
        self.low = bar.low
        self.close = bar.close
        self.open = bar.open
        self.volume += bar.volume

    def update_ewa(self, new_val, old_ewa, weight):
        return (new_val * weight) + (old_ewa * (1 - weight))


    def update_indicators(self, base_indicator):
        self.ewa_pr = self.update_ewa(self.mean_pr, base_indicator.ewa_pr, self.pr_ewa_weight)
        self.sma_pr = (base_indicator.sma_ttl_pr - base_indicator.sma_first_pr + self.mean_pr) / (base_indicator.sma_queue.qsize() + 1)
        self.pr_diff = self.ewa_pr - base_indicator.ewa_pr
        self.pr_diff_ewa = self.update_ewa(self.pr_diff, base_indicator.pr_diff_ewa, self.pr_diff_ewa_weight)
        mean_pr_diff = self.mean_pr - base_indicator.mean_pr
        self.up_ewa = self.update_ewa((0 if mean_pr_diff <= 0 else mean_pr_diff), base_indicator.up_ewa, self.rsi_weight)
        self.down_ewa = self.update_ewa((0 if mean_pr_diff >= 0 else -mean_pr_diff), base_indicator.down_ewa, self.rsi_weight)
        rs = (self.up_ewa / self.down_ewa) if self.down_ewa != 0 else 1.0
        self.rsi = 100 if self.down_ewa == 0 else 100.0 - (100.0 / (1 + rs))
        sto_max = max(self.high, base_indicator.sto_min_max_sw.max_val) # this section causes innacurate (or unexpected) sto calculations for weekends. Based on last bar which can be outside of time range
        sto_min = min(self.low, base_indicator.sto_min_max_sw.min_val) # can fix this by just using the current high and low if base indicator high and low is outside time range
        self.sto_high = sto_max
        self.sto_low = sto_min
        self.sto = (self.mean_pr - sto_min) / (sto_max - sto_min + self.eps) * 100 # note that mean is used here instead of close
        self.prev_close = base_indicator.close
        self.prev_high = base_indicator.high
        self.prev_low = base_indicator.low
        up_dir = max(self.high - self.prev_high, 0)
        down_dir = max(self.prev_low - self.low, 0)
        self.plus_dir_mov =  up_dir if up_dir > down_dir else 0
        self.min_dir_mov = down_dir if down_dir > up_dir else 0
        self.true_range = max(self.prev_close, self.high) - min(self.prev_close, self.low)
        self.fast_volume_ewa = self.update_ewa(self.volume, base_indicator.fast_volume_ewa, 1.0 / self.vol_fast_period)
        self.slow_volume_ewa = self.update_ewa(self.volume, base_indicator.slow_volume_ewa, 1.0 / self.vol_slow_period)
        self.volume_roc = (self.fast_volume_ewa - self.slow_volume_ewa) / (self.slow_volume_ewa + self.eps)
        self.volume_roc_last = base_indicator.volume_roc
        self.sma_volume_roc = (base_indicator.sma_ttl_vol_roc - base_indicator.sma_first_vol_roc + self.volume_roc) / (base_indicator.vol_roc_queue.qsize() + 1)
        self.sma_volume_roc_diff = (base_indicator.sma_ttl_vol_roc_diff - base_indicator.sma_first_vol_roc_diff + (self.sma_volume_roc - self.volume_roc ) ** 2) / (base_indicator.vol_roc_diff_queue.qsize() + 1)
        self.sma_volume_roc_diff_std = self.sma_volume_roc_diff ** (1 / 2.0)
        if base_indicator.first_tr_sum_cnt <= self.adx_period:
            self.first_tr_sum_cnt = base_indicator.first_tr_sum_cnt + 1
            self.true_range_sum = base_indicator.true_range_sum + self.true_range
            self.plus_dir_mov_sum = base_indicator.plus_dir_mov_sum + self.plus_dir_mov
            self.min_dir_mov_sum = base_indicator.min_dir_mov_sum + self.min_dir_mov
            self.true_range_sm = self.true_range_sum + self.eps
            self.plus_dir_mov_sm = self.plus_dir_mov_sum  + self.eps
            self.min_dir_mov_sm = self.min_dir_mov_sum  + self.eps
        else:
            self.true_range_sm = base_indicator.true_range_sm - (base_indicator.true_range_sm / self.adx_period) + self.true_range
            self.atr = self.true_range_sm / self.adx_period
            self.plus_dir_mov_sm = base_indicator.plus_dir_mov_sm - (base_indicator.plus_dir_mov_sm / self.adx_period) + self.plus_dir_mov
            self.min_dir_mov_sm = base_indicator.min_dir_mov_sm - (base_indicator.min_dir_mov_sm / self.adx_period) + self.min_dir_mov
            self.plus_dir_ind = self.plus_dir_mov_sm / self.true_range_sm * 100
            self.min_dir_ind = self.min_dir_mov_sm / self.true_range_sm * 100
            self.dir_mov_ind = abs(self.plus_dir_ind - self.min_dir_ind) / (self.plus_dir_ind + self.min_dir_ind + self.eps) * 100
            if base_indicator.first_dx_sum_cnt <= self.adx_period:
                self.first_dx_sum_cnt = base_indicator.first_dx_sum_cnt + 1.0
                self.dir_mov_ind_sum =  base_indicator.dir_mov_ind_sum + self.dir_mov_ind
                self.adx = self.dir_mov_ind_sum / self.first_dx_sum_cnt
            else:
                self.adx = ((base_indicator.adx * (self.adx_period - 1)) + self.dir_mov_ind) / self.adx_period

    def copy_indicators(self, temp_indicator, copy_time):
        self.high = temp_indicator.high
        self.low = temp_indicator.low
        self.close = temp_indicator.close
        self.ewa_pr = temp_indicator.ewa_pr
        self.sma_pr = temp_indicator.sma_pr
        self.sma_queue.put(temp_indicator.mean_pr)
        self.up_ewa = temp_indicator.up_ewa
        self.down_ewa = temp_indicator.down_ewa
        self.rsi = temp_indicator.rsi
        self.pr_diff_ewa = temp_indicator.pr_diff_ewa
        self.mean_pr = temp_indicator.mean_pr
        self.sma_ttl_pr += self.mean_pr - self.sma_first_pr
        self.sto = temp_indicator.sto
        self.sto_min_max_sw.update_min_max(self.high, self.low, copy_time)
        self.sto_high = self.sto_min_max_sw.max_val
        self.sto_low = self.sto_min_max_sw.min_val
        self.prev_close = temp_indicator.prev_close
        self.prev_high = temp_indicator.prev_high
        self.prev_low = temp_indicator.prev_low
        self.first_tr_sum_cnt = temp_indicator.first_tr_sum_cnt
        self.true_range_sum = temp_indicator.true_range_sum
        self.true_range_sm = temp_indicator.true_range_sm
        self.plus_dir_mov = temp_indicator.plus_dir_mov
        self.min_dir_mov = temp_indicator.min_dir_mov
        self.plus_dir_mov_sum = temp_indicator.plus_dir_mov_sum
        self.plus_dir_mov_sm = temp_indicator.plus_dir_mov_sm
        self.min_dir_mov_sum = temp_indicator.min_dir_mov_sum
        self.min_dir_mov_sm = temp_indicator.min_dir_mov_sm
        self.plus_dir_ind = temp_indicator.plus_dir_ind
        self.min_dir_ind = temp_indicator.min_dir_ind
        self.dir_mov_ind = temp_indicator.dir_mov_ind
        self.first_dx_sum_cnt = temp_indicator.first_dx_sum_cnt
        self.dir_mov_ind_sum = temp_indicator.dir_mov_ind_sum
        self.adx = temp_indicator.adx
        self.atr = temp_indicator.atr
        self.fast_volume_ewa = temp_indicator.fast_volume_ewa
        self.slow_volume_ewa = temp_indicator.slow_volume_ewa
        self.volume_roc = temp_indicator.volume_roc
        self.sma_volume_roc = temp_indicator.sma_volume_roc
        self.sma_volume_roc_diff = temp_indicator.sma_volume_roc_diff
        self.sma_ttl_vol_roc += self.volume_roc - self.sma_first_vol_roc
        self.vol_roc_queue.put(temp_indicator.volume_roc)
        self.sma_ttl_vol_roc_diff += ((self.sma_volume_roc - self.volume_roc ) ** 2) - self.sma_first_vol_roc_diff
        self.vol_roc_diff_queue.put((self.sma_volume_roc - self.volume_roc ) ** 2)
        self.sma_volume_roc_diff_std = temp_indicator.sma_volume_roc_diff_std
        self.volume = temp_indicator.volume
        self.volume_roc_last = temp_indicator.volume_roc_last


class OnlineIndicator(object):

    def __init__(self, sma_period=200, rsi_period=7, pr_ewa_period=4, pr_diff_ewa_period=5,
                 initialization_sec_thresh=200, **kwargs):
        self.base_indicator = IndicatorHolder(sma_period=sma_period, rsi_period=rsi_period, pr_ewa_period=pr_ewa_period,
                                              pr_diff_ewa_period=pr_diff_ewa_period, **kwargs)
        self.temp_indicator = IndicatorHolder(sma_period=sma_period, rsi_period=rsi_period, pr_ewa_period=pr_ewa_period,
                                              pr_diff_ewa_period=pr_diff_ewa_period, **kwargs)
        self.initialized = False
        self.initialization_sec_thresh = initialization_sec_thresh
        self.start_time = None


    def update_indicators(self, price, size, time, ohlc=False):
        update_price = price if not ohlc else price.close
        if not self.temp_indicator.initialized:
            self.start_time = time
            self.temp_indicator.initialize(update_price, size, time)
            self.base_indicator.copy_indicators(self.temp_indicator, time)
        else:
            if (time - self.start_time).total_seconds() >= self.initialization_sec_thresh:
                self.initialized = True
            if self.temp_indicator.current_time != time:
                self.base_indicator.copy_indicators(self.temp_indicator, self.temp_indicator.current_time)
                self.temp_indicator.initialize(update_price, size, time)
            if self.base_indicator.sma_queue.qsize() == self.temp_indicator.sma_period:
                self.base_indicator.sma_first_pr = self.base_indicator.sma_queue.get()
            if self.base_indicator.vol_roc_queue.qsize() == self.temp_indicator.vol_sma_period:
                self.base_indicator.sma_first_vol_roc = self.base_indicator.vol_roc_queue.get()
                self.base_indicator.sma_first_vol_roc_diff = self.base_indicator.vol_roc_diff_queue.get()
            if ohlc:
                self.temp_indicator.update_price_OHLC(price)
            else:
                self.temp_indicator.update_price(price, size)
            self.temp_indicator.update_indicators(self.base_indicator)

            

            

def app_control(func):
    def wrapper(*args, **kwargs):
        args[0]._App__close_application = kwargs['close_app']
        return func(*args, **kwargs)
    return wrapper

def app_respond(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        symbol = args[0]._App__req_id_to_sym[args[1]]
        print('ids ', args[0]._App__proc_req_ids)
        args[0]._App__proc_req_ids[symbol].remove(args[1])
        all_symbols_processed = np.all(list((len(args[0]._App__proc_req_ids[sym]) == 0) for sym in args[0]._App__proc_req_ids))

            if (len(args[0]._App__history_and_trade_req_id_params) > 0): # initiate trade after processing all initial history requests
                kw_dict = args[0]._App__history_and_trade_req_id_params[args[1]]
                print('checking len for ', symbol, len(args[0]._App__proc_req_ids[symbol]) == 0)
                if (len(args[0]._App__proc_req_ids[symbol]) == 0):    
                    print('creating dataframe for ', symbol)
                    args[0].historical_indicators_dfs[symbol] = pd.DataFrame(
                        args[0].historical_indicators[symbol],
                        columns=['date', *args[0].show_indicators]
                    )
                if  args[0]._App__trade_mode and all_symbols_processed and not args[0]._App__close_application :
                    start_time = datetime.datetime.now()
                    app_req_ids = args[0].trade_strategy(
                        start_time, **kw_dict
                    )
        condition = (args[0]._App__close_application == True)
        condition = condition and all_symbols_processed
        condition = condition and (not args[0]._App__trade_mode)
        if condition:
            super(App, args[0]).disconnect()
    return wrapper

class Wrapper(wrapper.EWrapper):
    def __init__(self):
        pass

class Client(client.EClient):
    def __init__(self, wrapper):
        client.EClient.__init__(self, wrapper)

### !!!!! Important!!! symbols to be traded are defined by symbols in online indicators
class App(Wrapper, Client):
    def __init__(self, entry_objs=None, online_indicators={},
                 pair_online_indicator=None,
                 show_indicators=['rsi', 'adx', 'sma_pr', 'pr_diff_ewa', 'sto'],
                 loss_timeout=30, win_timeout=30, trade_mode=True,
                 logging_dir='logging', trade_and_disconnect=False,
                 trade_loss_disconnect=True, loss_limit=3):
        Wrapper.__init__(self)
        Client.__init__(self, wrapper=self)
        self.logging_dir = logging_dir
        self.entry_objs = entry_objs
        self.online_indicators = online_indicators 
        self.symbols = list(self.online_indicators.keys())
        self.historical_data = {symbol: [] for symbol in self.symbols} # may need to add BACK a dictionary of request id
        self.pair_online_indicator = pair_online_indicator
        self.show_indicators = show_indicators
        self.loss_timeout = loss_timeout
        self.win_timeout = win_timeout
        self.timeout_tm = datetime.datetime.now()
        self.__trade_mode = trade_mode
        self.total_profit = 0
        self.__current_req_id = None
        self.accountsList = None
        self.__close_application = False
        self.__proc_req_ids = {symbol: [] for symbol in self.symbols}
        self.__sec_contract = None
        self.__stop_time = None
        self.__stop_limit = 10
        self.__realtime_iter = 0
        self.__tick_collect = False
        self.__unq_id = None
        self.__trade_strategy_bool = False
        self.__trade_strategy_cnt_thresh = 300
        self.__trade_and_disconnect = trade_and_disconnect
        self.__trade_loss_disconnect = trade_loss_disconnect
        self.__trade_loss_limit = loss_limit
        self.position_qty = 0
        self.next_valid_odr_id = 1
        self.pending_orders = set([])
        self.order_profit = {}
        self.win_loss = {'wins': 0, 'losses': 0}
        self.loss_streak = 0
        self.__clear_pending_set = set(['cancelled', 'filled'])
        self.parent_order_to_sig = {}
        self.profit_order_to_sig = {}
        self.stop_order_to_sig = {}
        self.__req_id_to_sym = {} 
        self.get_unique_id()
        self.timezone = timezone('US/Eastern')
        self.__history_and_trade_req_id_params = {}
        self.historical_indicators = {symbol: [] for symbol in self.symbols}
        self.historical_indicators_dfs = {symbol: None for symbol in self.symbols}

    def map_orders_to_sig(self, sig, bracket_odr):
        parent, takeProfit, stopLoss = bracket_odr
        self.parent_order_to_sig[parent.orderId] = sig
        self.profit_order_to_sig[takeProfit.orderId] = sig
        self.stop_order_to_sig[stopLoss.orderId] = sig

    def error(self, reqId: int, errorCode: int, errorString: str):
        super().error(reqId, errorCode, errorString)
        print("Error. Id:", reqId, "Code:", errorCode, "Msg:", errorString)

    def historicalData(self, reqId, bar):
        #print("HistoricalData. ReqId:", reqId, "BarData.", end='\r')
        self.historical_data[self.__req_id_to_sym[reqId]].append(bar)
        trade_tm_dt = datetime.datetime.strptime(
            bar.date[:-2] + '00', '%Y%m%d  %H:%M:%S'
        )
        reqId = str(reqId)
        year = int(reqId[:4])
        month = int(reqId[4:6])
        contract_year, contract_month = date_to_contract(trade_tm_dt.date())
        if (year == contract_year) and (month == contract_month):
            reqId = int(reqId)
            self.online_indicators[self.__req_id_to_sym[reqId]].update_indicators(
                bar, bar.volume, trade_tm_dt, ohlc=True
            )
            self.historical_indicators[self.__req_id_to_sym[reqId]].append(
                [bar.date] + [self.online_indicators[self.__req_id_to_sym[reqId]].temp_indicator.__dict__[ind]
                for ind in self.show_indicators]
            )


    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        print('next valid id: ', orderId)
        self.next_valid_odr_id = orderId

    def nextOrderId(self):
        self.next_valid_odr_id += 1
        return self.next_valid_odr_id

    def openOrder(self, orderId, contract, order, orderState):
        super().openOrder(orderId, contract, order, orderState)

    def track_stategy_performance(self, status, order_id):
        if order_id in self.parent_order_to_sig and status.lower() == 'filled':
            sig = self.parent_order_to_sig[order_id]
            if not self.entry_objs[sig].in_position:
                self.entry_objs[sig].entry(datetime.datetime.now())
        elif order_id in self.profit_order_to_sig and status.lower() == 'filled':
            sig = self.profit_order_to_sig[order_id]
            if self.entry_objs[sig].in_position:
                self.entry_objs[sig].exit(datetime.datetime.now(), 1)
        elif order_id in self.stop_order_to_sig and status.lower() == 'filled':
            sig = self.stop_order_to_sig[order_id]
            if self.entry_objs[sig].in_position:
                self.entry_objs[sig].exit(datetime.datetime.now(), -1)

    def orderStatus(self, orderId, status: str, filled: float,
                    remaining: float, avgFillPrice: float, permId: int,
                    parentId: int, lastFillPrice: float, clientId: int,
                    whyHeld: str, mktCapPrice: float):
        super().orderStatus(orderId, status, filled, remaining,
        avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice)
        self.track_stategy_performance(status, orderId)
        if status.lower() == 'filled' and parentId in self.order_profit:
            sign = 1 if self.order_profit[parentId][1] == 'BUY' else -1
            profit = (-sign * self.order_profit[parentId][0]) + (sign * avgFillPrice)
            if  self.order_profit[parentId][2] == 0: # protection against duplicate messages
                self.total_profit += profit
                if profit > 0:
                    self.win_loss['wins'] += 1
                    self.loss_streak = 0
                    self.timeout_tm = datetime.datetime.now() + datetime.timedelta(seconds=self.win_timeout)
                else:
                    self.win_loss['losses'] += 1
                    self.loss_streak += 1
                    self.timeout_tm = datetime.datetime.now() + datetime.timedelta(seconds=self.loss_timeout)
            self.order_profit[parentId][2] = profit
        if status.lower() in self.__clear_pending_set:
            if orderId in self.pending_orders:
                self.pending_orders.remove(orderId)
        if ((self.win_loss['losses'] - self.win_loss['wins']) >= self.__trade_loss_limit) and self.__trade_loss_disconnect:
            super().disconnect()

    def position(self, account: str, contract, pos: float,
                 avgCost: float):
        super().position(account, contract, pos, avgCost)
        '''
        print("Position.", "Account:", account, "Symbol:", contract.symbol, "SecType:",
              contract.secType, "Currency:", contract.currency,
              "Position:", pos, "Avg cost:", avgCost)
        '''
        self.position_qty = pos
        if self.__trade_and_disconnect and (self.position_qty != 0):
            super().disconnect()

    def positionEnd(self):
        super().positionEnd()
        #print("PositionEnd")

    @app_respond
    def historicalDataEnd(self, reqId: int, start: str, end: str):
        super().historicalDataEnd(reqId, start, end)
        print("HistoricalDataEnd. ReqId:", reqId, "from", start, "to", end)

    def tickByTickAllLast(self, reqId: int, tickType: int, time: int, price: float,
                      size: int, tickAtrribLast, exchange: str,
                      specialConditions: str):
        super().tickByTickAllLast(reqId, tickType, time, price, size, tickAtrribLast,
                                  exchange, specialConditions)
        self.__realtime_iter += 1
        if tickType == 1:
            print("Last.", end='\r')
        else:
            trade_tm_dt_full = datetime.datetime.fromtimestamp(time)
            trade_tm_dt = datetime.datetime(
                trade_tm_dt_full.year, trade_tm_dt_full.month, trade_tm_dt_full.day,
                trade_tm_dt_full.hour, trade_tm_dt_full.minute
            )
            trade_tm = trade_tm_dt.strftime("%Y%m%d %H:%M:%S")
            trade_tm_full = trade_tm_dt_full.strftime("%Y%m%d %H:%M:%S")
            indicators = []
            sig_performance = []
            trade_loss_tm_allow = self.timeout_tm < trade_tm_dt_full
            old_tm = self.online_indicator.temp_indicator.current_time
            self.online_indicator.update_indicators(price, size, trade_tm_dt)
            if trade_tm_dt != old_tm:
                self.historical_indicators.append(
                    [trade_tm] + [self.online_indicator.temp_indicator.__dict__[ind]
                    for ind in self.show_indicators]
                )
                self.historical_indicators_df = pd.DataFrame(
                    self.historical_indicators, columns=['date', *self.show_indicators]
                )
                #self.historical_indicators_df.to_parquet(f'{self.logging_dir}/streaming_data.parquet')
            if self.__trade_strategy_bool and self.__realtime_iter > self.__trade_strategy_cnt_thresh:
                if (self.position_qty == 0 and len(self.pending_orders) == 0):
                    for entry_obj_key in self.entry_objs:
                        entry_obj = self.entry_objs[entry_obj_key]
                        if entry_obj.enter(self.online_indicator) and trade_loss_tm_allow:
                            bracketOrder = self.place_order(
                                'MES', datetime.datetime.now(self.timezone), price + (entry_obj.entry_price_offset * entry_obj.sign),
                                entry_obj.action, transmit=True, order_timeout=entry_obj.trade_wait_tm,
                                profit_offset=entry_obj.profit_taker, stop_loss_offset=entry_obj.stop_loss,
                                quantity=entry_obj.quantity
                            )
                            self.map_orders_to_sig(entry_obj_key, bracketOrder)
                        sig_performance.extend(
                            ['sig name', entry_obj_key, 'wins', entry_obj.wins,
                             'losses', entry_obj.losses, 'ttl prof', entry_obj.cum_profit]
                        )
            for ind in self.show_indicators:
                indicators.extend([f'{ind}:', f'{self.online_indicator.temp_indicator.__dict__[ind]:.3f}'])
            print(" ReqId:", reqId, " Itter:", self.__realtime_iter,
                  "Time:", trade_tm_full,"Price:", price, "position:", self.position_qty,
                  'loss timeout:', not trade_loss_tm_allow, 'total profit:', self.total_profit,
                  'wins:', self.win_loss['wins'], 'losses:', self.win_loss['losses'], 'loss streak: ', self.loss_streak,
                  "open orders:", len(self.pending_orders), *indicators,  *sig_performance, end='\r')
            if self.__tick_collect: 
                # this will break if we collect symbols for more than 10 symbols
                self.historical_data[self.__req_id_to_sym[str(self.__current_req_id)[:9]]].append((datetime.datetime.fromtimestamp(time), price, size, datetime.datetime.now()))
            if self.__realtime_iter > self.__stop_limit and self.__tick_collect: # this appears to be hear for collecting realtime tick data and then exiting
                super().disconnect()



    def get_contract(self, symbol, secType='STK', currency='USD', exchange='SMART', futures_month=None):
        sec_contract = contract.Contract()
        sec_contract.includeExpired = True
        sec_contract.symbol = symbol
        sec_contract.secType = secType
        sec_contract.currency = currency
        sec_contract.exchange = exchange
        sec_contract.lastTradeDateOrContractMonth = futures_month
        return sec_contract

    def get_unique_id(self, filepath='counter.txt'):
        counter = 1
        if self.__unq_id is None:
            if not opath.exists(filepath):
                with open(filepath, 'w') as cnt_file:
                    cnt_file.write('1')
            else:
                with open(filepath, 'r') as cnt_file:
                    counter = int(cnt_file.read())
                with open(filepath, 'w') as cnt_file:
                    cnt_file.write(str(counter + 5))
        else:
            counter = self.__unq_id + 5
            self.__unq_id = counter
        return counter


    @app_control
    def get_historical_data_futures(self, exchange='GLOBEX',
                            history_len=5, history_unit='D', bar_unit='min', bar_length=1,
                            only_RTH=0, **kwargs):
        
        for symbol in self.symbols:
            req_ids = []
            days = [(datetime.datetime.now() - datetime.timedelta(days=day)).date() for day in range(history_len)]
            contracts = set([date_to_contract(day) for day in days])
            for contract in sorted(contracts):
                unq_id = self.get_unique_id()
                req_id = int(f'{contract[0]}{contract[1]:02}{unq_id}')
                self.__current_req_id = req_id
                print('symbol ', symbol, ' req id ', req_id)
                sec_contract = self.get_contract(symbol, secType='FUT', exchange=exchange,
                                                 futures_month=f'{contract[0]}{contract[1]:02}')
                sym_data = self.reqHistoricalData(req_id, sec_contract, "", f"{history_len} {history_unit}",
                                                  f"{bar_length} {bar_unit}", 'TRADES', only_RTH, 1, False, [])
                req_ids.append(req_id)
                self.__req_id_to_sym[req_id] = symbol
            self.__proc_req_ids[symbol] = copy.deepcopy(req_ids)
        return req_ids


    def trade_strategy(self, start_time, exchange='GLOBEX', limit=9000, tick_collect=False,
                       trade_strategy_cnt_thresh=300,
                       **kwargs):
        unq_id = self.get_unique_id()
        contract = date_to_contract(start_time.date())
        req_id = int(f'{contract[0]}{contract[1]:02}{unq_id}')
        self.__current_req_id = req_id
        for symbol in self.symbols:
            sec_contract = self.get_contract(symbol, secType='FUT', exchange=exchange,
                                             futures_month=f'{contract[0]}{contract[1]:02}')
            sym_data = self.reqTickByTickData(req_id, sec_contract, "AllLast", 0, True)
            self.__req_id_to_sym[req_id] = symbol
        self.get_positions()
        self.__stop_limit = limit
        self.__realtime_iter = 0
        self.__tick_collect = tick_collect
        self.__trade_strategy_bool = True
        self.__trade_strategy_cnt_thresh = trade_strategy_cnt_thresh
        return req_id

    def get_historical_and_trade_strategy_futs(self, exchange='GLOBEX',
                                          only_RTH=0, history_len=30, bar_length=1, bar_unit='min',
                                          close_app=False,
                                          **kwargs):

        req_ids = self.get_historical_data_futures(
            close_app=close_app, only_RTH=only_RTH, history_len=history_len,  bar_length=bar_length, bar_unit=bar_unit,
            secType='FUT', exchange=exchange
        )
        self.__history_and_trade_req_id_params = {
            req_id: dict(only_RTH=only_RTH,
                         exchange=exchange, **kwargs)
            for req_id in req_ids
        }


    def place_order(self, symbol, start_time, price, action, exchange='GLOBEX', quantity=1,
                     profit_offset=1, stop_loss_offset=1, transmit=False, order_timeout=5):
        contract = date_to_contract(start_time.date())
        active_stop_time = (start_time + datetime.timedelta(seconds=order_timeout)).strftime('%Y%m%d %H:%M:%S EST')
        print('current time ', datetime.datetime.now(self.timezone))
        print('stop time ', active_stop_time)
        sec_contract = self.get_contract(symbol, secType='FUT', exchange=exchange,
                                         futures_month=f'{contract[0]}{contract[1]:02}')
        parent = order.Order()
        parent.orderId = self.next_valid_odr_id
        parent.action = action
        parent.tif = "GTD"
        parent.orderType = "LMT"
        parent.totalQuantity = quantity
        parent.lmtPrice = price
        parent.transmit = False
        parent.goodTillDate = active_stop_time

        takeProfit = order.Order()
        takeProfit.orderId = parent.orderId + 1
        takeProfit.action = "SELL" if action == "BUY" else "BUY"
        takeProfit.orderType = "LMT"
        takeProfit.totalQuantity = quantity
        takeProfit.lmtPrice = price + profit_offset if action == "BUY" else price - profit_offset
        takeProfit.parentId = parent.orderId
        takeProfit.transmit = False

        stopLoss = order.Order()
        stopLoss.orderId = parent.orderId + 2
        stopLoss.action = "SELL" if action == "BUY" else "BUY"
        stopLoss.orderType = "STP"
        #Stop trigger price
        stopLoss.auxPrice = price - stop_loss_offset if action == "BUY" else price + stop_loss_offset
        stopLoss.totalQuantity = quantity
        stopLoss.parentId = parent.orderId
        stopLoss.outsideRth = True
        #In this case, the low side order will be the last child being sent. Therefore, it needs to set this attribute to True
        #to activate all its predecessors
        stopLoss.transmit = transmit

        bracketOrder = [parent, takeProfit, stopLoss]
        for o in bracketOrder:
            self.placeOrder(o.orderId, sec_contract, o)
            self.pending_orders.add(o.orderId)
            self.nextOrderId()
        self.order_profit[parent.orderId] = [price, action, 0]
        return bracketOrder

    def get_managed_accounts(self):
        self.reqManagedAccts()

    def managedAccounts(self, accountsList):
        print("got account list " + accountsList)
        self.accountsList = accountsList

    def get_positions(self):
        self.reqPositions()
