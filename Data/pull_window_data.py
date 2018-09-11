"""

Author: Zhou, Jincheng / Li, Yang

Pulling small-cap companies' data from WRDS within a fixed window period.

Small-cap companies are determined based on the calculation of market cap quantiles.
There are two types of data for each company instance:
    Financials data pulled each quater
    Monthly data pulled each month
The two types of data are stored independently.
Data for every company instance is guaranteed to be complete (some may still be 0.0 though)

About indexing in the stored dataset files:
    In the files for financials data, feature labels are concatenated with an integer
        indicating the number of quaters starting from the top year.
    In the files for monthly data, this integer indicates the number of months.

July 7th, 2018

"""

import wrds
import pandas as pd
import os
from pandas.tseries.offsets import MonthEnd
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


# --------------------------------- Custom Parameters ---------------------------------------------
# Custom parameters
top_year = 1984         # Top year : pull data starting at this year
bottom_year = 1993      # Bottom year : pull data until this year
window_size = 3         # Window size : each data instance of a company has window size of 3 years
num_quantiles = 5       # Number of quantiles : Companies in these first few number of quantiles
                        #                       are considered as small-cap companies
directory = "window_data" # Data directory : the datasets are stored in this directory
# -------------------------------------------------------------------------------------------------


# Code from prof. Neel. Return the first trading day in the given date of year
def first_trading_day(date):

    dow = date.dayofweek  # Monday = 0, Sunday = 6

    if dow in [5, 6]:
        date = date + pd.Timedelta(str(7 - dow) + ' days')

    if dow == 0:
        date = date + pd.Timedelta(str(1) + ' days')

    if date.year == 2007:  # Day of mourning for G . Ford on this year
        date = date + pd.Timedelta(str(1) + ' days')

    return date.strftime('%Y-%m-%d')


# -------------------------------------------------------------------------------------------------

# Connect to WRDS. Here the username and passcode would be asked for
db = wrds.Connection()

# Counter for # companies
num_companies_total = 0

# Iterate through each year to obtain a dataset per year
for year in range(top_year, bottom_year + 1):
    print("Year: %d" % year)

    # Calculate first trading date
    start_date = first_trading_day(pd.to_datetime('1-2-' + str(year)))

    # ---------------------------------------------------------------------------------------------
    #       phase 1: Obtain all companies' market cap information to find small-cap companies
    # ---------------------------------------------------------------------------------------------

    # Calculate time range for the first month
    start_date_dt = pd.to_datetime(start_date)
    date = [start_date_dt + MonthEnd(0) - pd.Timedelta(str(10) + ' days'), start_date_dt + MonthEnd(1)]
    date_str = [date[0].strftime('%Y-%m-%d'), date[1].strftime('%Y-%m-%d')]

    # SQL statement
    statement = "select all cusip, shrout, prc from crspq.msf \
                 where date > '{}' and date <= '{}'".format(date_str[0], date_str[1])

    # Pull data
    marketcap_df = db.raw_sql(statement)

    # Calculate market cap : Market Cap = Shrout * abs(prc)
    marketcap_df['market_cap'] = marketcap_df['shrout'] * abs(marketcap_df['prc'])

    # Sort company instances by market cap
    marketcap_df.sort_values(by=['market_cap'], inplace=True)

    # Delete all company instances with NaN market cap
    marketcap_df = marketcap_df[marketcap_df['market_cap'].notnull()]

    # Calculate cumulative sum of market cap
    marketcap_df['market_cap_cum'] = marketcap_df['market_cap'].cumsum()

    # Calculate quantile index for each company instance
    marketcap_df['quantile'] = marketcap_df['market_cap_cum'].transform(lambda x: pd.cut(x, 40, labels=range(1,41)))

    # Extract the cusip of all companies in the first few quantiles
    smallcap_df = marketcap_df.loc[marketcap_df['quantile'].isin(range(num_quantiles + 1))]
    smallcap_df.reset_index(drop=True, inplace=True)
    smallcap_cusip = smallcap_df[['cusip']]

    # ---------------------------------------------------------------------------------------------
    #       phase 2: Obtain all companies' financials from the 12 consecutive quaters
    # ---------------------------------------------------------------------------------------------

    # Initialize merged dataframe
    merged_fin_df = smallcap_cusip.copy()

    # Iterate through all quaters
    print("Pulling financial data")
    for y in range(year, year + window_size):
        for q in [1, 2, 3, 4]:
            print("\tyear: %d, quater: %d" % (y, q))

            # First trading day in each year
            yearly_start_date = first_trading_day(pd.to_datetime('1-2-' + str(y)))

            # SQL statement
            statement = "select a.cusip, b.conm, b.tic, b.rdq, b.atq, b.dlttq, b.ltq, b.cheq, \
                         b.oibdpq, b.ppentq, b.piq, b.revtq, b.niq, b.cogsq, b.xoprq, b.xintq, \
                         b.oiadpq, b.dlcq from crspq.dsf a join comp.fundq b on b.cusip::varchar(8) = a.cusip \
                         where a.date = '{}' and b.fyearq = {} and b.fqtr = {}" \
                        .format(yearly_start_date, y, q)

            # Pull data
            financials_qtr_df = db.raw_sql(statement)

            # Create the label mapping to rename column labels
            label_map = {x: x + "_" + str(q + (y - year) * 4) for x in financials_qtr_df.columns if
                         x not in ['cusip', 'conm', 'tic']}

            # Use label mapping to rename column labels
            financials_qtr_df.rename(columns=label_map, inplace=True)

            # Merge dataframe according to values in 'cusip' column
            if 'conm' and 'tic' not in merged_fin_df.columns:
                merged_fin_df = pd.merge(merged_fin_df, financials_qtr_df, how='inner', on=['cusip'])
            else:
                merged_fin_df = pd.merge(merged_fin_df, financials_qtr_df, how='inner', on=['cusip', 'conm', 'tic'])

    # Eliminate all NaN data
    merged_fin_df_clean = merged_fin_df.copy()

    for column in merged_fin_df_clean.columns:
        if column not in ['conm', 'tic']:
            merged_fin_df_clean = merged_fin_df_clean.loc[merged_fin_df_clean[column].notnull()]

    merged_fin_df_clean.reset_index(drop=True, inplace=True)

    # ---------------------------------------------------------------------------------------------
    #       phase 3: Obtain all companies' monthly data from the 36 consecutive months
    # ---------------------------------------------------------------------------------------------

    # Initialize merged dataframe
    merged_monthly_df = smallcap_cusip.copy()

    # Iterate through all months
    print("\nPulling monthly data")
    for m in range(12 * window_size):
        print("\tMonth: %d" % m)

        # Initialize time range for each month
        start_date_dt = pd.to_datetime(start_date)
        date = [start_date_dt + MonthEnd(m + 1) - pd.Timedelta(str(10) + ' days'),
                start_date_dt + MonthEnd(m + 1)]
        date_str = [date[0].strftime('%Y-%m-%d'), date[1].strftime('%Y-%m-%d')]

        # SQL statement
        statement = "select all cusip, vol, ret, bid, ask, shrout, prc from crspq.msf \
                     where date > '{}' and date <= '{}'".format(date_str[0], date_str[1])

        # Pull data
        monthly_data_df = db.raw_sql(statement)

        # Create label mapping to rename column labels
        label_map = {x: x + "_" + str(m + 1) for x in monthly_data_df.columns if x not in ['cusip']}

        # Use label mapping to rename column labels
        monthly_data_df.rename(columns=label_map, inplace=True)

        # Merge dataframe according to values in 'cusip' column
        merged_monthly_df = pd.merge(merged_monthly_df, monthly_data_df, how='inner', on=['cusip'])

    # Eliminate all NaN data
    merged_monthly_df_clean = merged_monthly_df.copy()

    for column in merged_monthly_df_clean.columns:
        merged_monthly_df_clean = merged_monthly_df_clean.loc[merged_monthly_df_clean[column].notnull()]

    merged_monthly_df_clean.reset_index(drop=True, inplace=True)

    # ---------------------------------------------------------------------------------------------
    #       phase 4: Select companies in both financial and monthly dataframes based on common cusip
    # ---------------------------------------------------------------------------------------------

    # Get cusip index from both dataframes
    fin_cusip_index = pd.Index(merged_fin_df_clean['cusip'])
    monthly_cusip_index = pd.Index(merged_monthly_df_clean['cusip'])

    # Intersection on cusip index
    common_cusip_index = fin_cusip_index.intersection(monthly_cusip_index)

    # Get filtered common financial and monthly dataframes
    joint_fin_df = merged_fin_df_clean.loc[merged_fin_df_clean['cusip'].isin(common_cusip_index)]
    joint_fin_df.reset_index(drop=True, inplace=True)
    joint_monthly_df = merged_monthly_df_clean.loc[merged_monthly_df_clean['cusip'].isin(common_cusip_index)]
    joint_monthly_df.reset_index(drop=True, inplace=True)

    # ---------------------------------------------------------------------------------------------
    #       phase 5: Store as datasets and print summaries
    # ---------------------------------------------------------------------------------------------

    # Store datasets
    if not os.path.isdir(directory):
        os.mkdir(directory)

    financials_file = "financials_%d.csv" % year
    monthly_file = "monthly_%d.csv" % year
    fin_path = os.path.join(directory, financials_file)
    mon_path = os.path.join(directory, monthly_file)

    joint_fin_df.to_csv(fin_path)
    joint_monthly_df.to_csv(mon_path)

    # Print summaries
    num_companies = common_cusip_index.shape[0]
    num_companies_total += num_companies
    print("\t### %d companies in year %d ###" % (num_companies, year))

# Print total number of companies
print("\n##### %d company instances pulled in total #####" % num_companies_total)
