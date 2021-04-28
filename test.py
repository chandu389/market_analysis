#!/usr/bin/env python3
"""

"""
__author__ = "G V R"
__credits__ = ["nse tools"]
__version__ = "0.0.1"

import argparse
import math
import sys
import requests
from pprint import pprint
import logging
from nsetools import Nse
from tabulate import tabulate
import time
from datetime import date, datetime
from MaxHeap import MaxHeap
import locale


class test():
    def __init__(self):

        self.RETURN_PERCENT = {
            'SAFE': 12,
            'MODERATE': 18,
            'AGGRESSIVE': 24
        }
        self.COOKIE_URL = 'https://www.nseindia.com/get-quotes/derivatives?symbol=BANKNIFTY'
        self.API_URL = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        self.MARGIN_URL = "https://zerodha.com/margin-calculator/SPAN"
        self.NIFTY_INDEX = "nifty 50"
        self.MARGIN_REQ_STRANGLE = 150000
        self.NIFTY_LOT_SIZE = 75
        self.WEEKLY = 7
        self.MONTHLY = 30
        self.BIMONTHLY = 60
        self.TRIMONTHLY = 90
        self.YEAR_DAYS = 360
        self.EXPIRY_DATE_FORMAT = '%d-%b-%Y'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36'
        }
        self.near_expiry = None
        self.near_month_expiry = None
        self.next_month_expiry = None
        self.far_month_expiry = None
        self.near_expiry_date = None
        self.far_expiry_date = None
        self.nse = Nse()
        self.desc = "Algorithmic trading"
        parser = argparse.ArgumentParser(description=self.desc)
        parser.add_argument('-m', '--mode',
                            choices=['SAFE', 'MODERATE', 'AGGRESSIVE'], default='SAFE',
                            help="The mode of return SAFE, MODERATE, AGGRESSIVE")
        parser.add_argument('-f', '--function',
                            choices=['OI', 'RETURNS', 'EXPIRYDATES'], default='RETURNS',
                            help="The mode of return SAFE, MODERATE, AGGRESSIVE")
        parser.add_argument('-s', '--strategy',
                            choices=['STRANGLE', 'STRADDLE', 'BULLCALLSPREAD', 'BEARCALLSPREAD', 'BEARPUTSPREAD', 'BULLPUTSPREAD'], default='STRANGLE',
                            help="The strategy to calculate for")
        parser.add_argument('-l', '--log-level',
                            choices=['DEBUG', 'INFO', 'WARNING'], default=logging.INFO,
                            help="Set the log level for standalone logging")
        parser.add_argument('-e', '--expiry_date', default=None,
                            help="Calculate returns for specific expiry dates")
        parser.add_argument('-d', '--market_depth', default=5,
                            help="Depth/Number of options to be shown")
        args = parser.parse_args()
        self.args = args
        self.parser = parser
        self.strategy = args.strategy
        self.market_depth = args.market_depth
        self.vix = self.nse.get_index_quote('INDIA VIX')['lastPrice']
        self.data = self.get_req_data(self.get_api_url)

        # Initiate expiry dates
        self.expiry_dates = self.data['records']['expiryDates']
        self.near_expiry_date = self.expiry_dates[0]
        self.far_expiry_date = self.expiry_dates[-1]
        self.expiry_date = self.near_expiry_date if (self.args.expiry_date is None) else self.args.expiry_date
        self.get_period_details(self.data)
        self.get_current_nifty_index = self.get_atm_price()
        margin = self.margin_calculator(
            strikes=[int(self.get_current_nifty_index) + 1500, int(self.get_current_nifty_index) - 1500],
            expiry_dates=[self.zerodha_expiry(self.near_month_expiry), self.zerodha_expiry(self.near_month_expiry)],
            option_types=['CE', 'PE'], qty=[self.NIFTY_LOT_SIZE, self.NIFTY_LOT_SIZE],
            trade=['sell', 'sell'])

        self.cutoff_price = self.get_price(self.args.strategy, self.args.mode, self.NIFTY_LOT_SIZE, margin, self.WEEKLY)
        self.cutoff_price = self.get_price(self.args.strategy, self.args.mode, self.NIFTY_LOT_SIZE, margin, self.WEEKLY)
        self.cutoff_price = self.get_price(self.args.strategy, self.args.mode, self.NIFTY_LOT_SIZE, margin, self.WEEKLY)

    @property
    def get_nifty_index(self):
        return self.NIFTY_INDEX

    def get_return(self, flag):
        if flag in self.RETURN_PERCENT:
            return self.RETURN_PERCENT[flag]
        else:
            return None

    @property
    def get_api_url(self):
        return self.API_URL

    def get_index_around_price(self, price, flag):
        # records.data[0].PE.lastPrice
        data = self.requester(self.API_URL)

    def requester(self, url):
        with requests.session() as s:
            # load cookies:
            s.get(self.COOKIE_URL, headers=self.headers)

            # get data:
            data = s.get(url, headers=self.headers).json()
            # print(json.dumps(data, indent=4))
            return data
            # print data to screen:

    def get_atm_price(self):
        nifty_json = self.nse.get_index_quote(self.NIFTY_INDEX)
        last_price = nifty_json['lastPrice']
        atm_price = round(last_price, -2)
        return atm_price

    def get_open_interest(self, data, expiry_date):
        s1 = time.time()
        ce_oi = []
        pe_oi = []
        size = self.market_depth
        oi_ce_heap = MaxHeap(len(data))
        oi_pe_heap = MaxHeap(len(data))

        for row in data:
            if row['expiryDate'] == expiry_date:
                if 'CE' in row:
                    oi_ce_heap.insert(
                        [row['strikePrice'], row['CE']['openInterest'], row['CE']['changeinOpenInterest']])
                if 'PE' in row:
                    oi_pe_heap.insert(
                        [row['strikePrice'], row['PE']['openInterest'], row['PE']['changeinOpenInterest']])

        for idx in range(self.market_depth):
            ce_oi.append(oi_ce_heap.extractMax())
            pe_oi.append(oi_pe_heap.extractMax())

        e1 = time.time()

        s2 = time.time()
        tmp1 = [[float('-inf'), 0, 0] for i in range(size)]
        for row in data:
            start = 0
            end = size - 1
            if row['expiryDate'] == expiry_date:
                if 'CE' in row:
                    op_int = row['CE']['openInterest']
                    i = 0
                    while i < size:
                        if tmp1[i][0] < op_int:
                            tmp1.insert(i, [op_int, row['CE']['changeinOpenInterest'], row['strikePrice']])
                            # tmp.insert(i, op_int)
                            break
                        else:
                            i += 1
                elif 'PE' in row:
                    pass

        e2 = time.time()
        # print(f'time till execution mid - {e1 - s1}')
        # print(f'time till execution mid - {e2 - s2}')

        print(tabulate(ce_oi, headers=['ce-strikePrice', 'OI', 'Change in OI', ],
                       tablefmt='pretty'))
        print(tabulate(pe_oi, headers=['pe-strikePrice', 'OI', 'Change in OI', ],
                       tablefmt='pretty'))

    def get_req_data(self, url):
        data = self.requester(url)
        return data

    def get_expiry_date(self):
        expiry_dates = self.data['records']['expiryDates']
        print(tabulate([[item] for item in expiry_dates], headers=['Expiry-Dates'],
                       tablefmt='pretty'))

    def returns(self):
        pe_count = 0
        ce_count = 0
        res_arr = []
        lowest_pe_price = sys.maxsize
        lowest_ce_price = sys.maxsize
        lowest_pe_strike_price = 0
        lowest_ce_strike_price = 0

        for row in self.data['records']['data']:
            # print(row['strikePrice'])
            if 'PE' in row and pe_count < self.market_depth and row['PE']['expiryDate'] == self.expiry_date and row['PE'][
                'lastPrice'] != 0:
                # Finding only far away price irrespective of whatever price may be. Only once
                if lowest_pe_price == 0:
                    lowest_pe_price = row['PE']['lastPrice']
                    lowest_pe_strike_price = row['strikePrice']

                if float(row['PE']['lastPrice']) > self.cutoff_price:
                    if len(res_arr) <= pe_count:
                        res_arr.append({})
                    dict = res_arr[pe_count]
                    dict['expiry'] = row['PE']['expiryDate']
                    dict['pe'] = 'PE'
                    dict['pe-strike'] = row['strikePrice']
                    dict['pe-price'] = row['PE']['lastPrice']
                    pe_count += 1

            if 'CE' in row and ce_count < self.market_depth and row['CE']['expiryDate'] == self.expiry_date and row['CE'][
                'lastPrice'] != 0:

                # Finding only far away price irrespective of whatever price may be
                lowest_ce_price = row['CE']['lastPrice']
                lowest_ce_strike_price = row['strikePrice']

                if float(row['CE']['lastPrice']) < self.cutoff_price:
                    if len(res_arr) <= ce_count:
                        res_arr.append({})
                    dict = res_arr[ce_count]
                    dict['ce'] = 'CE'
                    dict['ce-strike'] = row['strikePrice']
                    dict['ce-price'] = row['CE']['lastPrice']
                    ce_count += 1

        while pe_count < self.market_depth:
            res_arr[pe_count]['pe'] = 'PE'
            res_arr[pe_count]['pe-strike'] = lowest_pe_strike_price
            res_arr[pe_count]['pe-price'] = lowest_pe_price
            pe_count += 1

        while ce_count < self.market_depth:
            res_arr[ce_count]['ce'] = 'ce'
            res_arr[ce_count]['ce-strike'] = lowest_ce_strike_price
            res_arr[ce_count]['ce-price'] = lowest_ce_price
            ce_count += 1

        res_dict_arr = [[dict['expiry'], dict['ce'], dict['ce-strike'], dict['ce-price'], dict['pe'], dict['pe-strike'],
                         dict['pe-price']] for dict in res_arr]

        print(tabulate(res_dict_arr, headers=['Expiry', 'type', 'strikePrice', 'Price', 'type', 'strikePrice', 'Price'],
                       tablefmt='pretty'))

        # Evaluate return rates
        # locale.setlocale(locale.LC_ALL, '')
        expiry = self.zerodha_expiry(self.near_month_expiry)
        least_margin = self.margin_calculator(strikes=[res_arr[self.market_depth - 1]['ce-strike'], res_arr[0]['pe-strike']],
                                              expiry_dates=[expiry, expiry],
                                              option_types=['CE', 'PE'], qty=[self.NIFTY_LOT_SIZE, self.NIFTY_LOT_SIZE],
                                              trade=['sell', 'sell'])
        least_return_rate = self.return_calc([res_arr[self.market_depth - 1]['ce-price'], res_arr[0]['pe-price']],
                                             strategy=self.args.strategy, lot_size=self.NIFTY_LOT_SIZE,
                                             expiry_date=self.expiry_date, margin=least_margin)
        best_margin = self.margin_calculator(strikes=[res_arr[0]['ce-strike'], res_arr[self.market_depth - 1]['pe-strike']],
                                             expiry_dates=[expiry, expiry],
                                             option_types=['CE', 'PE'], qty=[self.NIFTY_LOT_SIZE, self.NIFTY_LOT_SIZE],
                                             trade=['sell', 'sell'])
        best_return_rate = self.return_calc([res_arr[self.market_depth - 1]['pe-price'], res_arr[0]['ce-price']],
                                            strategy=self.args.strategy, lot_size=self.NIFTY_LOT_SIZE,
                                            expiry_date=self.expiry_date, margin=best_margin)
        return_scenarios = [
            ['Least Return', round(least_return_rate, 2), '{:,.2f}'.format(least_margin),
             res_arr[self.market_depth - 1]['ce-strike'],
             res_arr[self.market_depth - 1]['ce-price'], res_arr[0]['pe-strike'], res_arr[0]['pe-price']],
            ['Best Return', round(best_return_rate, 2), f'{best_margin:,.2f}', res_arr[0]['ce-strike'],
             res_arr[0]['ce-price'],
             res_arr[self.market_depth - 1]['pe-strike'], res_arr[self.market_depth - 1]['pe-price']]
        ]
        print(tabulate(return_scenarios,
                       headers=['Return Type', 'Return Rate', 'Margin', 'CE strike', 'CE strikeprice', 'PE strike',
                                'PE strikePrice'],
                       tablefmt='pretty'))

    def oi_analysis(self):
        self.get_open_interest(self.data['records']['data'], self.expiry_date)

    def return_calc(self, prices, margin=None, strategy=None, lot_size=None, days=None, expiry_date=None, strikes=None):
        if expiry_date:
            delta_days = self.get_delta_days(expiry_date)
        elif days:
            delta_days = days

        if strategy == 'STRANGLE':
            return_percentage = (sum(prices) * self.YEAR_DAYS * lot_size * 100) / (float(margin) * delta_days)

        return return_percentage

    def zerodha_expiry(self, expiry):
        month = expiry.split('-')[1]
        year = expiry.split('-')[2]
        return "NIFTY" + str(int(year) % 100) + month.upper()

    def get_price(self, strategy, flag, lot_size, margin, frequency):
        # Returns formula in % : r = (price * lotsize / Margin) * (360/Days) * 100
        percent = self.get_return(flag)
        delta_days = self.get_delta_days(self.expiry_date)
        # TO DO : implement for remaining strategies
        price = None
        if strategy == 'STRANGLE':
            price = margin * percent * delta_days / (self.YEAR_DAYS * 100 * 2 * lot_size)
        return price

    def get_delta_days(self, expiry_date):
        return (datetime.strptime(expiry_date, self.EXPIRY_DATE_FORMAT).date() - date.today()).days + 1

    def get_period_details(self, data):
        dates = data['records']['expiryDates']
        self.near_expiry = datetime.strptime(dates[0], self.EXPIRY_DATE_FORMAT)
        near_date = self.near_expiry.date()
        near_month = near_date.month
        near_month_year = near_date.year
        next_month_year = far_month_year = near_month_year
        next_month = (near_month % 12) + 1
        far_month = ((near_month + 1) % 12) + 1
        if near_month == 11:
            next_month_year = near_month_year
            far_month_year = near_month_year + 1
        elif near_month == 12:
            next_month_year = near_month_year + 1
            far_month_year = near_month_year + 1

        self.near_month_expiry = self.get_expiry_dates(dates, near_month, near_month_year)
        self.next_month_expiry = self.get_expiry_dates(dates, next_month, next_month_year)
        self.far_month_expiry = self.get_expiry_dates(dates, far_month, far_month_year)

    def get_expiry_dates(self, dates, month, year):
        prev_date = dates[0]
        for expiry_date in dates:
            expiry_date_obj = datetime.strptime(expiry_date, self.EXPIRY_DATE_FORMAT).date()
            if expiry_date_obj.month <= month and expiry_date_obj.year <= year:
                prev_date = expiry_date
            else:
                monthly_expiry = prev_date
                break
        return monthly_expiry

    def margin_calculator(self, strikes=None, expiry_dates=None, option_types=None, qty=None, trade=None):
        payload = {
            'action': 'calculate',
            'exchange[]': ['NFO', 'NFO'],
            'product[]': ['OPT', 'OPT'],
            'scrip[]': expiry_dates,
            'option_type[]': option_types,
            'strike_price[]': strikes,
            'qty[]': qty,
            'trade[]': trade
        }
        session = requests.Session()
        res = session.post(self.MARGIN_URL, data=payload)
        data = res.json()
        if data is not None:
            return data['total']['total']


# main function
if __name__ == "__main__":
    ins = test()
    if ins.args.function == 'EXPIRYDATES':
        ins.get_expiry_date()
    elif ins.args.function == 'OI':
        ins.oi_analysis()
    elif ins.args.function == 'RETURNS':
        print(tabulate(['STRATEGY DEPLOYED - ' + ins.strategy]))
        print(tabulate(['BREAK EVEN PRICE FOR A LOT - {:.2f}'.format(ins.cutoff_price)]))
        ins.returns()

    # start = time.time()
    # end = time.time()
    # print(f'time till execution mid - {mid - start}')
