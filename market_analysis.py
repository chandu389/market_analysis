#!/usr/bin/env python3
"""

"""
__author__ = "G V R"
__credits__ = ["nse tools"]
__version__ = "0.0.1"

import argparse
import argcomplete
import math
import sys
import httpx
import requests
from pprint import pprint
import logging
from tabulate import tabulate
import time
import datetime
from MaxHeap import MaxHeap
import pandas as pd
from deprecated import deprecated
import asyncio
from aiohttp import ClientSession
import httpcore
import json

class MarketAnalysis():
    def __init__(self):
        start = time.time()
        self.RETURN_PERCENT = {
            'SAFE': 12,
            'MODERATE': 18,
            'AGGRESSIVE': 24
        }
        self.COOKIE_URL = 'https://www.nseindia.com/get-quotes/derivatives?symbol=BANKNIFTY'
        self.API_URL = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        self.MARGIN_URL = "https://zerodha.com/margin-calculator/SPAN"
        self.SGB_URL = "https://www1.nseindia.com/live_market/dynaContent/live_watch/stock_watch/sovGoldStockWatch.json"
        self.SGB_BOND_URL = "https://www.nseindia.com/api/quote-equity"
        self.HISTORICAL_DATA_URL = "https://www.nseindia.com/api/historical/indicesHistory"
        self.INDICES_URL = "https://www.nseindia.com/api/equity-master"
        self.MARGIN = {
            'STRANGLE': 100000,
            'BULLCALLSPREAD': 22000
        }
        self.NIFTY_INDEX = "nifty 50"
        self.MARGIN_REQ_STRANGLE = 150000
        self.NIFTY_LOT_SIZE = 50
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
        self.market_depth = 5
        self.nse_session = None
        self.historical_data = None
        self.nse_client_session = None
        self.desc = "Algorithmic trading"
        parser = argparse.ArgumentParser(description=self.desc, formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('command', help="""command to execute a function like <oi> , <returns>, <expirydates> , <hr>

For hr command, the following indices are supported:
+----------------------------+--------------------------------+---------------------------+---------------------------------+--------------------+
|    Broad Market Indices    |        Sectoral Indices        |     Thematic Indices      |        Strategy Indices         |       Others       |
+----------------------------+--------------------------------+---------------------------+---------------------------------+--------------------+
|          NIFTY 50          |           NIFTY AUTO           |     NIFTY COMMODITIES     | NIFTY DIVIDEND OPPORTUNITIES 50 | Securities in F&O  |
|       NIFTY NEXT 50        |           NIFTY BANK           |  NIFTY INDIA CONSUMPTION  |        NIFTY50 VALUE 20         | Permitted to Trade |
|      NIFTY MIDCAP 50       |          NIFTY ENERGY          |        NIFTY CPSE         |       NIFTY100 QUALITY 30       |                    |
|      NIFTY MIDCAP 100      |    NIFTY FINANCIAL SERVICES    |   NIFTY INFRASTRUCTURE    |      NIFTY50 EQUAL WEIGHT       |                    |
|      NIFTY MIDCAP 150      | NIFTY FINANCIAL SERVICES 25/50 |         NIFTY MNC         |      NIFTY100 EQUAL WEIGHT      |                    |
|     NIFTY SMALLCAP 50      |           NIFTY FMCG           |  NIFTY GROWTH SECTORS 15  |   NIFTY100 LOW VOLATILITY 30    |                    |
|     NIFTY SMALLCAP 100     |            NIFTY IT            |         NIFTY PSE         |         NIFTY ALPHA 50          |                    |
|     NIFTY SMALLCAP 250     |          NIFTY MEDIA           |   NIFTY SERVICES SECTOR   |       NIFTY200 QUALITY 30       |                    |
|   NIFTY MIDSMALLCAP 400    |          NIFTY METAL           |    NIFTY100 LIQUID 15     |  NIFTY ALPHA LOW-VOLATILITY 30  |                    |
|         NIFTY 100          |          NIFTY PHARMA          |  NIFTY MIDCAP LIQUID 15   |      NIFTY200 MOMENTUM 30       |                    |
|         NIFTY 200          |         NIFTY PSU BANK         |    NIFTY INDIA DIGITAL    |   NIFTY MIDCAP150 QUALITY 50    |                    |
| NIFTY500 MULTICAP 50:25:25 |          NIFTY REALTY          |       NIFTY100 ESG        |                                 |                    |
|   NIFTY LARGEMIDCAP 250    |       NIFTY PRIVATE BANK       | NIFTY INDIA MANUFACTURING |                                 |                    |
|    NIFTY MIDCAP SELECT     |     NIFTY HEALTHCARE INDEX     |                           |                                 |                    |
|     NIFTY TOTAL MARKET     |    NIFTY CONSUMER DURABLES     |                           |                                 |                    |
|     NIFTY MICROCAP 250     |        NIFTY OIL & GAS         |                           |                                 |                    |
+----------------------------+--------------------------------+---------------------------+---------------------------------+--------------------+
                            """)
        parser.add_argument('-l', '--log-level',
                            choices=['DEBUG', 'INFO', 'WARNING'], default=logging.INFO,
                            help="Set the log level for standalone logging")
        argcomplete.autocomplete(parser)
        args = parser.parse_args(sys.argv[1:2])
        e1 = time.time()
        # print(f'time till execution e1 - {e1 - start}')
        self.args = args
        e2 = time.time()
        # print(f'time till execution e2 - {e2 - start}')
        #self.vix = self.nse.get_index_quote('INDIA VIX')['lastPrice']
        e3 = time.time()
        # print(f'time till execution e3 - {e3 - start}')
        self.data = self.get_req_data(self.get_api_url)
        e4 = time.time()
        # print(f'time till execution e4 - {e4 - start}')

        # Initiate expiry dates
        self.expiry_dates = self.data['records']['expiryDates']
        self.near_expiry_date = self.expiry_dates[0]
        self.far_expiry_date = self.expiry_dates[-1]
        self.expiry_date = self.near_expiry_date
        self.get_current_nifty_index = self.data['records']['underlyingValue']
        

        e5 = time.time()
        # print(f'time till execution e5 - {e5 - start}')
        self.get_period_details(self.data)
        e6 = time.time()
        # print(f'time till execution e6 - {e6 - start}')
        e7 = time.time()
        # print(f'time till execution e7 - {e7 - start}')
        if not hasattr(self, args.command):
            logging.error("{} is not valid command".format(args.command))
            sys.exit(1)

        getattr(self, args.command)()

    def initiate_nse_session(self):
        self.nse_session = requests.Session()
        self.nse_session.get(self.COOKIE_URL, headers=self.headers)
        
    def indices(self):
        self.get_indices()
    
    def get_indices(self):
        data = self.requester(self.INDICES_URL)
        headers = [key for key in data.keys()]
        max_len = max([len(data[key]) for key in data.keys()])
        indices = [[] for i in range(max_len)]
        for idx in range(max_len):
            for key in data.keys():
                if idx < len(data[key]):
                    indices[idx].append(data[key][idx])
                else:
                    indices[idx].append("")
        print(tabulate(indices, headers=headers, tablefmt='pretty', showindex=False))
        #print(json.dumps(data, indent=4))
        
       
    def oi(self):
        """
          Analyse Open Interest
        """
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('-e', '--expiry_date', default=None,
                            help="expiry date for options")
        parser.add_argument('-d', '--market_depth', default=5,
                            help="Depth/Number of options to be shown")
        args = parser.parse_args(sys.argv[2:])
        self.market_depth = int(args.market_depth)
        if args.expiry_date is not None:
            self.expiry_date = args.expiry_date
        self._oi_analysis()

    def sgb(self):
        """
        :returns
        """
        self._sgb_analysis()

    def _sgb_analysis(self):
        sgb_start = time.time()
        self.sgb_data = self.get_req_data(self.get_sgb_api_url)
        df = pd.json_normalize(self.sgb_data, record_path=['data'])
        df['qty'] = df['qty'].astype(int)
        df['ltP'] = df['ltP'].str.replace(",", "").astype(float)
        issue_prices = [0 for i in range(len(self.sgb_data["data"]))]
        interest_rates = [0 for i in range(len(self.sgb_data["data"]))]
        issue_dates = [0 for i in range(len(self.sgb_data["data"]))]

        # with ThreadPoolExecutor(max_workers=60) as executor:
        #     for idx, row in enumerate(self.sgb_data["data"]):
        #         executor.submit(self.get_x, row, idx, issue_prices, interest_rates, issue_dates)

        async def async_run():
            async with httpx.AsyncClient() as client:
                await client.get(self.COOKIE_URL, headers=self.headers)
                data = await asyncio.gather(
                    *(client.get(self.SGB_BOND_URL, params={'symbol': row['symbol']}, headers=self.headers) for
                      idx, row in enumerate(self.sgb_data["data"])))
            for idx, row in enumerate(data):
                # data = await client.get(self.SGB_BOND_URL, params={'symbol': row['symbol']}, headers=self.headers)
                sgb_bond_data = row.json()
                issue_prices[idx] = sgb_bond_data["securityInfo"]["faceValue"]
                interest_rates[idx] = sgb_bond_data["info"]["companyName"]
                issue_dates[idx] = sgb_bond_data["metadata"]["listingDate"]

        asyncio.run(async_run())

        df['issue_price'] = issue_prices
        df['interest_rate'] = interest_rates
        df['issue_date'] = issue_dates

        df.sort_values("qty", inplace=True, ascending=False)

        print(tabulate(df[['symbol', 'issue_price', 'issue_date', 'interest_rate', 'ltP', 'chn', 'qty']],
                       headers=['Symbol', 'Issue Price', 'Issue Date', 'Interest Rate', 'LTP', 'Change',
                                'Quantity Traded'],
                       tablefmt='pretty', showindex=False))
        sgb_end = time.time()
        print(f'time till execution - {sgb_end - sgb_start}')

    def get_x(self, row, idx, issue_prices, interest_rates, issue_dates):
        print(idx)
        sgb_bond_data = self.requester(self.SGB_BOND_URL, params={'symbol': row['symbol']})
        issue_prices[idx] = sgb_bond_data["securityInfo"]["faceValue"]
        interest_rates[idx] = sgb_bond_data["info"]["companyName"]
        issue_dates[idx] = sgb_bond_data["metadata"]["listingDate"]
        print(idx)

    def returns(self):
        """
            :returns
        """
        parser = argparse.ArgumentParser(description='Argument parser for returns command')
        parser.add_argument('-m', '--mode',
                            choices=['SAFE', 'MODERATE', 'AGGRESSIVE'], default='SAFE',
                            help="The mode of return SAFE, MODERATE, AGGRESSIVE")
        parser.add_argument('-s', '--strategy',
                            choices=['STRANGLE', 'STRADDLE', 'BULLCALLSPREAD', 'BEARCALLSPREAD', 'BEARPUTSPREAD',
                                     'BULLPUTSPREAD'], default='STRANGLE',
                            help="The strategy to calculate for")
        parser.add_argument('-e', '--expiry_date', default=None,
                            help="Calculate returns for specific expiry dates")
        parser.add_argument('-d', '--market_depth', default=5,
                            help="Depth/Number of options to be shown")
        args = parser.parse_args(sys.argv[2:])
        self.strategy = args.strategy
        self.mode = args.mode
        if args.expiry_date is not None:
            self.expiry_date = args.expiry_date
        self.market_depth = args.market_depth

        self._get_returns()

    def expirydates(self):
        self.get_expiry_data()

    def hr(self):
        """
        :returns
        """
        # Argument parser for historical returns
        parser = argparse.ArgumentParser(description='Argument parser for hisrotical returns command')
        parser.add_argument('-i', '--index_type', default=None,
                            help="index type for historical data. For example NIFTY MIDCAP 150, NIFTY SMALLCAP 250")
        parser.add_argument('-r', '--frequency', default="Yearly", type=str, choices=["Monthly", "Yearly"],
                            help="Frequency for historical data. For example Monthly, Yearly")
        parser.add_argument('-t', '--to_date', default=None, type=lambda s: datetime.datetime.strptime(s, '%d-%m-%Y'),
                            help="to date for historical data. Format dd-mm-yyyy (example 31-01-2022")
        parser.add_argument('-f', '--from_date', default=None, type=lambda s: datetime.datetime.strptime(s, '%d-%m-%Y'),
                            help="to date for historical data. Format dd-mm-yyyy (example 31-01-2022")
        args = parser.parse_args(sys.argv[2:])
        index_type = args.index_type
        
        asyncio.run(self.fetch_historical_data(self.HISTORICAL_DATA_URL, index_type, args.from_date, args.to_date))
        hr_df = pd.json_normalize(self.historical_data, record_path=['data', 'indexCloseOnlineRecords'])
        if hr_df.empty:
            print("No data found for the given date range")
            exit(0)
        hr_df['EOD_TIMESTAMP'] = pd.to_datetime(hr_df['EOD_TIMESTAMP'])
        hr_df = hr_df.sort_values(by='EOD_TIMESTAMP')
        start_date = max (args.from_date, hr_df.loc[1, ["EOD_TIMESTAMP"]].values[0]) # start date should be the first available date in the data but since we need for one previous day as well we take second entry
        if start_date != args.from_date:
            print(f"Start date is adjusted to the first available date {start_date.strftime('%d-%m-%Y')}")
        self.calculate_historical_returns(hr_df, args.frequency, start_date, args.to_date)
        
    def calculate_historical_returns(self, data, frequency='Yearly', start_date=None, end_date=None):
        results = []
        if frequency == 'Yearly':
            while start_date <= end_date:
                end_cur_date = min (end_date,  datetime.datetime(start_date.year, 12, 31, 0, 0, 0))
                results.append([start_date.year, self.calculate_returns(data, start_date, end_cur_date)])
                start_date = end_cur_date + datetime.timedelta(days=1)
            print(tabulate(results, headers=['Year', 'Change %', ], tablefmt='pretty', showindex=False))
            
        elif frequency == 'Monthly':
            while start_date <= end_date:
                end_cur_date = min (end_date,  pd.to_datetime(start_date).to_period('M').end_time)
                results.append([start_date.year, start_date.strftime("%b"), self.calculate_returns(data, start_date, end_cur_date)])
                start_date = end_cur_date + datetime.timedelta(days=1)
            print(tabulate(results, headers=['Year', 'Month', 'Change %', ], tablefmt='pretty', showindex=False))
        else:
            pass
        
    def calculate_returns(self, data, start_date, end_date):
        cur_date = start_date + datetime.timedelta(days=-1)
        cur_price_data = data[data['EOD_TIMESTAMP'] == cur_date]
        if cur_price_data.empty:
            cur_price_data = data.iloc[data['EOD_TIMESTAMP'].searchsorted(cur_date) - 1]
            cur_price = cur_price_data['EOD_CLOSE_INDEX_VAL']
        else:
            cur_price = cur_price_data['EOD_CLOSE_INDEX_VAL'].values[0]
        
        end_price_data = data[data['EOD_TIMESTAMP'] == end_date]
        if end_price_data.empty:
            end_price_data = data.iloc[data['EOD_TIMESTAMP'].searchsorted(end_date) - 1]
            end_price = end_price_data['EOD_CLOSE_INDEX_VAL']
        else:
            end_price = end_price_data['EOD_CLOSE_INDEX_VAL'].values[0]
        
        return "{:.2f}".format(((end_price - cur_price) / cur_price) * 100)
    
    async def fetch_historical_data(self, url, index_type, start_date, end_date):
        tasks = []
        # Iterate through each year in the date range
        current_date = start_date + datetime.timedelta(days=-7)
        while current_date <= end_date:

            # Calculate the appropriate end date for this iteration
            end_date_for_iteration = min(current_date + datetime.timedelta(days=365), end_date)  # Ensure end date doesn't exceed overall end_date

            # Construct URL with start date and calculated end date
            start_date_str, end_date_str = self.format_hr_dates(current_date, end_date_for_iteration)
            
            # Create a task for each year's data fetch
            task = asyncio.create_task(self.async_requester(url, params={'indexType': index_type, 'from': start_date_str, 'to': end_date_str}))
            tasks.append(task)

            # Move to the next year
            current_date = end_date_for_iteration + datetime.timedelta(days=1)
            
        # Wait for all tasks to complete
        self.historical_data = await asyncio.gather(*tasks)
            
    def format_hr_dates(self, from_date, to_date):
        from_date = from_date.strftime('%d-%m-%Y')
        to_date = to_date.strftime('%d-%m-%Y')
        return from_date, to_date
        
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

    @property
    def get_sgb_api_url(self):
        return self.SGB_URL

    def get_index_around_price(self, price, flag):
        # records.data[0].PE.lastPrice
        data = self.requester(self.API_URL)

    def requester(self, url, params=None):
        if self.nse_session is None:
            self.initiate_nse_session()
        try:
            data = self.nse_session.get(url, params=params, headers=self.headers).json()
        except requests.exceptions.RequestException as e:
            self.nse_session.close()  # close the session
            self.initiate_nse_session()
            data = self.nse_session.get(url, params=params, headers=self.headers).json()
        
        #print(json.dumps(data, indent=4))
        return data

    async def async_requester(self, url, params=None):
        if self.nse_session is None:
            self.initiate_nse_session()
        try:
            data = self.nse_session.get(url, params=params, headers=self.headers).json()
        except requests.exceptions.RequestException as e:
            self.nse_session.close()  # close the session
            self.initiate_nse_session()
            data = self.nse_session.get(url, params=params, headers=self.headers).json()
        
        #print(json.dumps(data, indent=4))
        return data
        # if self.nse_client_session is None:
        #     print("chandra")
        #     self.nse_client_session = ClientSession()
        #     await self.nse_client_session.get(self.COOKIE_URL, headers=self.headers)
        #     print("end")
        # try:
        #     print(params)
        #     data = await self.nse_client_session.get(url, params=params, headers=self.headers)
        # except httpcore.ConnectError as e:
        #     self.nse_client_session.close()
        #     self.nse_client_session = ClientSession()
        #     data = await self.nse_client_session.get(url, params=params,headers=self.headers)
                
        # print(json.dumps(data, indent=4))
        # return data.json()

    def get_atm_price(self):
        nifty_json = self.nse.get_index_quote(self.NIFTY_INDEX)
        last_price = nifty_json['lastPrice']
        atm_price = round(last_price, -2)
        return atm_price

    @deprecated
    def get_open_interest(self, data, expiry_date):
        # s1 = time.time()
        # ce_oi = []
        # pe_oi = []
        # size = self.market_depth
        # oi_ce_heap = MaxHeap(len(data))
        # oi_pe_heap = MaxHeap(len(data))
        #
        # for row in data:
        #     if row['expiryDate'] == expiry_date:
        #         if 'CE' in row:
        #             oi_ce_heap.insert(
        #                 [row['strikePrice'], row['CE']['openInterest'], row['CE']['changeinOpenInterest']])
        #         if 'PE' in row:
        #             oi_pe_heap.insert(
        #                 [row['strikePrice'], row['PE']['openInterest'], row['PE']['changeinOpenInterest']])
        #
        # for idx in range(self.market_depth):
        #     ce_oi.append(oi_ce_heap.extractMax())
        #     pe_oi.append(oi_pe_heap.extractMax())
        #
        # e1 = time.time()
        #
        # s2 = time.time()
        # tmp1 = [[float('-inf'), 0, 0] for i in range(size)]
        # for row in data:
        #     start = 0
        #     end = size - 1
        #     if row['expiryDate'] == expiry_date:
        #         if 'CE' in row:
        #             op_int = row['CE']['openInterest']
        #             i = 0
        #             while i < size:
        #                 if tmp1[i][0] < op_int:
        #                     tmp1.insert(i, [op_int, row['CE']['changeinOpenInterest'], row['strikePrice']])
        #                     # tmp.insert(i, op_int)
        #                     break
        #                 else:
        #                     i += 1
        #         elif 'PE' in row:
        #             pass
        #
        # e2 = time.time()
        # print(f'time till execution without pandas - {e2 - s1}')
        # # print(f'time till execution mid - {e2 - s2}')
        #
        # print(tabulate(ce_oi, headers=['ce-strikePrice', 'OI', 'Change in OI', ],
        #                tablefmt='pretty'))
        # print(tabulate(pe_oi, headers=['pe-strikePrice', 'OI', 'Change in OI', ],
        #                tablefmt='pretty'))

        pd1 = time.time()
        df = pd.json_normalize(self.data, record_path=['records', 'data'])
        df.query("expiryDate=='{}'".format(self.expiry_date), inplace=True)
        final_ce_df = df[df['CE.strikePrice'].notnull()].sort_values("CE.openInterest", ascending=False).head(
            self.market_depth)
        final_pe_df = df[df['PE.strikePrice'].notnull()].sort_values("PE.openInterest", ascending=False).head(
            self.market_depth)

        pd2 = time.time()
        print(f'time  execution pandas - {pd2 - pd1}')

        print(tabulate(final_ce_df[['CE.strikePrice', 'CE.openInterest', 'CE.changeinOpenInterest']].astype('Int64'),
                       headers=['ce-strikePrice', 'OI', 'Change in OI', ],
                       tablefmt='pretty', showindex=False))
        print(tabulate(final_pe_df[['PE.strikePrice', 'PE.openInterest', 'PE.changeinOpenInterest']].astype('Int64'),
                       headers=['pe-strikePrice', 'OI', 'Change in OI', ],
                       tablefmt='pretty', showindex=False))

    def get_req_data(self, url, params=None):
        data = self.requester(url, params=params)
        return data

    def get_expiry_data(self):
        expiry_dates = self.data['records']['expiryDates']
        print(tabulate([[item] for item in expiry_dates], headers=['Expiry-Dates'],
                       tablefmt='pretty'))

    def _get_returns(self):
        print(tabulate(['STRATEGY DEPLOYED - ' + self.strategy]))
        if self.strategy == 'STRANGLE':
            self.margin = self.margin_calculator(
                strikes=[int(self.get_current_nifty_index) + 1500, int(self.get_current_nifty_index) - 1500],
                expiry_dates=[self.zerodha_expiry(self.near_month_expiry), self.zerodha_expiry(self.near_month_expiry)],
                option_types=['CE', 'PE'], qty=[self.NIFTY_LOT_SIZE, self.NIFTY_LOT_SIZE],
                trade=['sell', 'sell'])
            self.cutoff_price = self.get_price(self.strategy, self.mode, self.NIFTY_LOT_SIZE, self.margin)
            print(tabulate(['BREAK EVEN PRICE FOR A LOT - {:.2f}'.format(self.cutoff_price)]))
            pe_count = 0
            ce_count = 0
            res_arr = []
            lowest_pe_price = sys.maxsize
            lowest_ce_price = sys.maxsize
            lowest_pe_strike_price = 0
            lowest_ce_strike_price = 0

            for row in self.data['records']['data']:
                # print(row['strikePrice'])
                if 'PE' in row and pe_count < self.market_depth and row['PE']['expiryDate'] == self.expiry_date and \
                        row['PE']['lastPrice'] != 0 and int(row['PE']['strikePrice']) % 100 == 0:
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

                if 'CE' in row and ce_count < self.market_depth and row['CE']['expiryDate'] == self.expiry_date and \
                        row['CE']['lastPrice'] != 0 and int(row['CE']['strikePrice']) % 100 == 0:

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

            res_dict_arr = [
                [dict['expiry'], dict['ce'], dict['ce-strike'], dict['ce-price'], dict['pe'], dict['pe-strike'],
                 dict['pe-price']] for dict in res_arr]

            print(tabulate(res_dict_arr,
                           headers=['Expiry', 'type', 'strikePrice', 'Price', 'type', 'strikePrice', 'Price'],
                           tablefmt='pretty'))

            # Evaluate return rates
            # locale.setlocale(locale.LC_ALL, '')
            expiry = self.zerodha_expiry(self.near_month_expiry)
            least_margin = self.margin_calculator(
                strikes=[res_arr[self.market_depth - 1]['ce-strike'], res_arr[0]['pe-strike']],
                expiry_dates=[expiry, expiry],
                option_types=['CE', 'PE'], qty=[self.NIFTY_LOT_SIZE, self.NIFTY_LOT_SIZE],
                trade=['sell', 'sell'])
            least_return_rate = self.return_calc([res_arr[self.market_depth - 1]['ce-price'], res_arr[0]['pe-price']],
                                                 strategy=self.strategy, lot_size=self.NIFTY_LOT_SIZE,
                                                 expiry_date=self.expiry_date, margin=least_margin)
            best_margin = self.margin_calculator(
                strikes=[res_arr[0]['ce-strike'], res_arr[self.market_depth - 1]['pe-strike']],
                expiry_dates=[expiry, expiry],
                option_types=['CE', 'PE'], qty=[self.NIFTY_LOT_SIZE, self.NIFTY_LOT_SIZE],
                trade=['sell', 'sell'])
            best_return_rate = self.return_calc([res_arr[self.market_depth - 1]['pe-price'], res_arr[0]['ce-price']],
                                                strategy=self.strategy, lot_size=self.NIFTY_LOT_SIZE,
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
        if self.strategy == 'BULLCALLSPREAD':
            default_round_arg = 100
            nearest_index = self.get_nearest_index(default_round_arg)
            interval_ = 100
            times_ = int(self.market_depth)
            margin = self.margin_calculator(
                strikes=[int(nearest_index), int(nearest_index) + interval_],
                expiry_dates=[self.zerodha_expiry(self.expiry_date), self.zerodha_expiry(self.expiry_date)],
                option_types=['CE', 'CE'], qty=[self.NIFTY_LOT_SIZE, self.NIFTY_LOT_SIZE],
                trade=['buy', 'sell'])
            # pd.options.display.float_format = "{:,.2f}".format
            df = pd.json_normalize(self.data, record_path=['records', 'data'])

            # Call options
            sd_df = df[
                (df['CE.strikePrice'].notnull()) &
                (df['expiryDate'] == self.expiry_date) &
                (df['strikePrice'] >= nearest_index - (interval_ * times_)) &
                (df['strikePrice'] <= nearest_index + (interval_ * times_)) &
                (df['strikePrice'] % 100 == 0)
                ][['CE.strikePrice', 'CE.lastPrice']] \
                .astype('float').sort_values("CE.strikePrice", ascending=False)

            range = [0]
            max_profit = [0]
            max_loss = [0]
            margin_arr = [margin]
            profit_per_lot = [0]
            loss_per_lot = [0]
            for index, row in sd_df.iterrows():
                real_index = sd_df.index.get_loc(index)
                if real_index > 0:
                    range.append('B-' + str(sd_df.iloc[real_index]['CE.strikePrice']) + '-S-' + str(
                        sd_df.iloc[real_index - 1]['CE.strikePrice']))
                    max_profit.append(round(interval_ - (
                            sd_df.iloc[real_index]['CE.lastPrice'] - sd_df.iloc[real_index - 1]['CE.lastPrice']),
                                            2))
                    max_loss.append(
                        round(sd_df.iloc[real_index - 1]['CE.lastPrice'] - sd_df.iloc[real_index]['CE.lastPrice'], 2))
                    margin_arr.append(margin)
                    profit_per_lot.append(round(max_profit[-1] * self.NIFTY_LOT_SIZE, 2))
                    loss_per_lot.append(round(max_loss[-1] * self.NIFTY_LOT_SIZE, 2))
            sd_df['range'] = range
            sd_df['max-profit'] = max_profit
            sd_df['max-loss'] = max_loss
            sd_df['margin'] = margin_arr
            sd_df['max-profit-per-lot'] = profit_per_lot
            sd_df['max-loss-per-lot'] = loss_per_lot
            print(
                tabulate(sd_df[['CE.strikePrice', 'CE.lastPrice', 'range', 'margin', 'max-profit', 'max-profit-per-lot',
                                'max-loss', 'max-loss-per-lot']][1:],
                         headers=['ce-strikePrice', 'ce-price', 'strategy', 'margin', 'max-profit',
                                  'max-profit-per-lot', 'max-loss', 'max-loss-per-lot'],
                         tablefmt='pretty', showindex=False))

            # put options
            pe_df = df[
                (df['PE.strikePrice'].notnull()) &
                (df['expiryDate'] == self.expiry_date) &
                (df['strikePrice'] >= nearest_index - (interval_ * times_)) &
                (df['strikePrice'] <= nearest_index + (interval_ * times_)) &
                (df['strikePrice'] % 100 == 0)
                ][['PE.strikePrice', 'PE.lastPrice']] \
                .astype('float').sort_values("PE.strikePrice", ascending=False)

            range = [0]
            max_profit = [0]
            max_loss = [0]
            margin_arr = [margin]
            profit_per_lot = [0]
            loss_per_lot = [0]

            for index, row in pe_df.iterrows():
                real_index = pe_df.index.get_loc(index)
                if real_index > 0:
                    range.append('S-' + str(pe_df.iloc[real_index - 1]['PE.strikePrice']) + '-' + 'B-' + str(
                        pe_df.iloc[real_index]['PE.strikePrice']))
                    max_loss.append(round((pe_df.iloc[real_index - 1]['PE.lastPrice'] - pe_df.iloc[real_index][
                        'PE.lastPrice']) - interval_, 2))
                    max_profit.append(
                        round(pe_df.iloc[real_index - 1]['PE.lastPrice'] - pe_df.iloc[real_index]['PE.lastPrice'], 2))
                    margin_arr.append(margin)
                    profit_per_lot.append(round(max_profit[-1] * self.NIFTY_LOT_SIZE, 2))
                    loss_per_lot.append(round(max_loss[-1] * self.NIFTY_LOT_SIZE, 2))
            pe_df['range'] = range
            pe_df['max-profit'] = max_profit
            pe_df['max-loss'] = max_loss
            pe_df['margin'] = margin_arr
            pe_df['max-profit-per-lot'] = profit_per_lot
            pe_df['max-loss-per-lot'] = loss_per_lot
            print(
                tabulate(pe_df[['PE.strikePrice', 'PE.lastPrice', 'range', 'margin', 'max-profit', 'max-profit-per-lot',
                                'max-loss', 'max-loss-per-lot']][1:],
                         headers=['pe-strikePrice', 'pe-price', 'strategy', 'margin', 'max-profit',
                                  'max-profit-per-lot', 'max-loss', 'max-loss-per-lot'],
                         tablefmt='pretty', showindex=False))

        if self.strategy == 'BEARCALLSPREAD':
            pass

        if self.strategy == 'STRADDLE':
            default_round_arg = 100
            nearest_index = self.get_nearest_index(default_round_arg)
            interval_ = 100
            times_ = int(self.market_depth)
            margin = self.margin_calculator(
                strikes=[int(nearest_index), int(nearest_index)],
                expiry_dates=[self.zerodha_expiry(self.expiry_date), self.zerodha_expiry(self.expiry_date)],
                option_types=['CE', 'PE'], qty=[self.NIFTY_LOT_SIZE, self.NIFTY_LOT_SIZE],
                trade=['sell', 'sell'])
            # pd.options.display.float_format = "{:,.2f}".format
            df = pd.json_normalize(self.data, record_path=['records', 'data'])

            # Call options
            sd_df = df[
                (df['expiryDate'] == self.expiry_date) &
                (df['strikePrice'] >= nearest_index - (interval_ * times_)) &
                (df['strikePrice'] <= nearest_index + (interval_ * times_)) &
                (df['strikePrice'] % 100 == 0)
                ][['CE.strikePrice', 'CE.lastPrice', 'PE.lastPrice']] \
                .astype('float').sort_values("CE.strikePrice", ascending=False)
            # print(sd_df)
            range = [0]
            max_profit = [0]
            max_loss = [0]
            margin_arr = [margin]
            profit_per_lot = [0]
            loss_per_lot = [0]
            for index, row in sd_df.iterrows():
                real_index = sd_df.index.get_loc(index)
                if real_index > 0:
                    # range.append('S-' + str(sd_df.iloc[real_index]['CE.strikePrice']) + '-S-' + str(
                    #     sd_df.iloc[real_index - 1]['CE.strikePrice']))
                    max_profit.append(
                        round(sd_df.iloc[real_index]['CE.lastPrice'] + sd_df.iloc[real_index]['PE.lastPrice'], 2))
                    margin_arr.append(margin)
                    profit_per_lot.append(round(max_profit[-1] * self.NIFTY_LOT_SIZE, 2))

            # sd_df['range'] = range
            sd_df['max-profit'] = max_profit
            sd_df['margin'] = margin_arr
            sd_df['max-profit-per-lot'] = profit_per_lot
            print(
                tabulate(sd_df[['CE.strikePrice', 'CE.lastPrice', 'PE.lastPrice', 'margin', 'max-profit',
                                'max-profit-per-lot',
                                ]][1:],
                         headers=['strikePrice', 'ce-price', 'pe-price', 'margin', 'max-profit',
                                  'max-profit-per-lot'],
                         tablefmt='pretty', showindex=False))

    def get_nearest_index(self, place):
        '''
         Returns nearest place index
        '''
        current_index = self.get_current_nifty_index
        if int(math.ceil(current_index / place)) * place - current_index > current_index - int(
                math.floor(current_index / place)) * place:
            nearest_index = int(math.ceil(current_index / place)) * place
        else:
            nearest_index = int(math.floor(current_index / place)) * place
        return nearest_index

    def _oi_analysis(self):
        pd1 = time.time()
        df = pd.json_normalize(self.data, record_path=['records', 'data'])
        df.query("expiryDate=='{}'".format(self.expiry_date), inplace=True)
        final_ce_df = df[df['CE.strikePrice'].notnull()].sort_values("CE.openInterest", ascending=False).head(
            self.market_depth)
        final_pe_df = df[df['PE.strikePrice'].notnull()].sort_values("PE.openInterest", ascending=False).head(
            self.market_depth)

        pd2 = time.time()
        # print(f'time  execution pandas - {pd2 - pd1}')

        print(tabulate(final_ce_df[['CE.strikePrice', 'CE.openInterest', 'CE.changeinOpenInterest']].astype('Int64'),
                       headers=['ce-strikePrice', 'OI', 'Change in OI', ],
                       tablefmt='pretty', showindex=False))
        print(tabulate(final_pe_df[['PE.strikePrice', 'PE.openInterest', 'PE.changeinOpenInterest']].astype('Int64'),
                       headers=['pe-strikePrice', 'OI', 'Change in OI', ],
                       tablefmt='pretty', showindex=False))

    def return_calc(self, prices, margin=None, strategy=None, lot_size=None, days=None, expiry_date=None, strikes=None):
        '''
        Calculates return percentage
        :param prices:
        :param margin:
        :param strategy:
        :param lot_size:
        :param days:
        :param expiry_date:
        :param strikes:
        :return:
        '''
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

    def get_price(self, strategy, flag, lot_size, margin):
        '''
        Calculates cut off price for a strategy based on mode

        :param strategy: STRANGLE, STRADDLE
        :param flag: mode of strategy aggressive, safe etc..
        :param lot_size:
        :param margin: margin amount required for strategy
        :return: Returns cut-off price to earn min percentage profit based on mode

        Returns formula in % : r = (price * lotsize / Margin) * (360/Days) * 100
        '''

        percent = self.get_return(flag)
        delta_days = self.get_delta_days(self.expiry_date)
        # TO DO : implement for remaining strategies
        cut_off_price = None
        if strategy == 'STRANGLE':
            cut_off_price = margin * percent * delta_days / (self.YEAR_DAYS * 100 * 2 * lot_size)
        return cut_off_price

    def get_delta_days(self, expiry_date):
        return (datetime.strptime(expiry_date, self.EXPIRY_DATE_FORMAT).date() - date.today()).days + 1

    def get_period_details(self, data):
        dates = data['records']['expiryDates']
        self.near_expiry = datetime.datetime.strptime(dates[0], self.EXPIRY_DATE_FORMAT)
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
            expiry_date_obj = datetime.datetime.strptime(expiry_date, self.EXPIRY_DATE_FORMAT).date()
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
            try:
                return data['total']['total']
            except:
                logging.warning('Margin calculation didnt work with api')
                return self.MARGIN[self.strategy]


# main function
if __name__ == "__main__":
    MarketAnalysis()
