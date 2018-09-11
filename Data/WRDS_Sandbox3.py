# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 18:16:05 2018

@author: Neel Tiruviluamala
"""

import wrds
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from pandas.tseries.offsets import MonthEnd
import locale
locale.setlocale( locale.LC_ALL, 'en_US.UTF-8' )


def first_trading_day(date): #If anyone can think of a better way of doing this, by all means implement it
    dow = date.dayofweek #Monday = 0, Sunday = 6
                            
    if dow in [5, 6]:
        date = date + pd.Timedelta( str(7 - dow) + ' days')
            
    if dow == 0:
        date = date + pd.Timedelta(str(1) + ' days')
    
    if date.year == 2007: #Day of mourning for G . Ford on this year
        date = date + pd.Timedelta(str(1) + ' days')
    
    return date.strftime('%Y-%m-%d')
    
db = wrds.Connection() #Enter user name and password.  Only Yifan should do this.

all_df = pd.DataFrame()

top_year = 1976
bottom_year = 1974

years = range(bottom_year, top_year, 1)

#for year, market_cap in zip(years,market_caps):
for year in years:
        ### Start_date is the first trading date of the year
        start_date = first_trading_day(pd.to_datetime('1-2-' + str(year))) 
            
        ### Set up a quarters dictionary
        l_year = year - 1
        ll_year = year - 2
        lll_year = year - 3
        
        quarters_index = {l_year: [3,2,1], ll_year: [4,3,2,1], lll_year: [4]}
        ###
        
        ### Set up the shell for the data
        statement = "select date, cusip from crspq.dsf where date = '" + start_date_str + "'"
        
        data = db.raw_sql(statement)
        data.set_index('cusip', inplace = True)
        ###
        
        ### Layer on financials 
        for y, y_label in zip([l_year, ll_year, lll_year], ['l_year', 'll_year', 'lll_year']):
            for q in quarters_index[y]:
                statement = "select a.cusip, b.conm, b.tic, b.rdq, b.atq, b.dlttq, b.ltq, b.cheq, b.oibdpq, b.ppentq, b.piq, b.revtq, b.niq, b.cogsq, b.xoprq, b.xintq, b.oiadpq, b.dlcq from crspq.dsf a join comp.fundq b on b.cusip::varchar(8) = a.cusip where a.date = '" + start_date_str + "' and b.fyearq = " + str(y) + " and b.fqtr = " + str(q) 
                data_layer = db.raw_sql(statement)
                
                data_layer.set_index('cusip', inplace = True)
                data_layer = data_layer[~data_layer.rdq.isnull()]
                
                conm = data_layer.conm
                data_layer.drop('conm',axis = 1, inplace = True)
                
                tic = data_layer.tic
                data_layer.drop('tic',axis = 1, inplace = True)
        
                
                new_cols = [old_col + "_" + y_label + "_" + str(q) for old_col in data_layer.columns]
                data_layer.columns = new_cols
                
                data = pd.merge(data_layer,data, how = 'inner', left_index = True, right_index = True)
                
                if 'conm' not in data.columns:
                    data = data.join(conm)
                
                if 'tic' not in data.columns:
                    data = data.join(tic)
        
        ### Layer on Monthly Data
        
        start = time.time()
        
        for i in range(48):
            
            start_date = pd.to_datetime(start_date)
            date = [start_date + MonthEnd(i+1),start_date + MonthEnd(i+1) - pd.Timedelta(str(10) + ' days')] 
            date_str = [date[0].strftime('%Y-%m-%d'),date[1].strftime('%Y-%m-%d')]
            statement = "select all cusip, vol, ret, bid, ask, shrout, prc from crspq.msf where date > '" + date_str[1] + "'" " and date <= '" + date_str[0] + "'"
                                                                                                              
            data_ti = db.raw_sql(statement)
            data_ti.set_index('cusip', inplace=True)
            
            new_cols = [old_col + "_" + str(i+1) for old_col in data_ti.columns]
            data_ti.columns = new_cols
            
            if not data_ti.empty: 
                data = data.join(data_ti)
        
            print(year, i, time.time() - start)
        ###
        
        data['year'] = year
        data.set_index([data.index.values, 'year'], inplace = True)
        all_df = all_df.append(data)
        all_df.reset_index().to_csv('all_df.csv', compression = 'gzip')

all_df['Market_Cap'] = all_df.shrout_1*abs(all_df.prc_1)

df = all_df.copy()[all_df.Market_Cap.notnull()]
df = df.reset_index().sort_values(by = ['year','Market_Cap']).set_index(['level_0','year'])
df['Market_Cap_Cum'] = df.groupby('year').Market_Cap.cumsum()
df['Market_Cap_Quantile'] = df.groupby(['year'])['Market_Cap_Cum'].transform(lambda x: pd.cut(x, 40, labels=range(1,41)))
