# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.2
#   kernelspec:
#     display_name: Python [conda env:bbytes] *
#     language: python
#     name: conda-env-bbytes-py
# ---

# +
import csv 
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from pathlib import Path

import streamlit as st
import plotly.express as px
import altair as alt
import dateutil.parser
import copy


# +
@st.experimental_memo
def get_hist_info(df_coin, principal_balance,plheader):
    numtrades = int(len(df_coin))
    numwin = int(sum(df_coin[plheader] > 0))
    numloss = int(sum(df_coin[plheader] < 0))
    winrate = int(np.round(100*numwin/numtrades,2))
    
    grosswin = sum(df_coin[df_coin[plheader] > 0][plheader])
    grossloss = sum(df_coin[df_coin[plheader] < 0][plheader])
    if grossloss !=0:
        pfactor = -1*np.round(grosswin/grossloss,2)
    else: 
        pfactor = np.nan
    return numtrades, numwin, numloss, winrate, pfactor
@st.experimental_memo
def get_rolling_stats(df, lev, otimeheader, days):
    rollend = datetime.today()-timedelta(days=days)
    rolling_df = df[df[otimeheader] >= rollend]

    if len(rolling_df) > 0:
        rolling_perc = rolling_df['Return Per Trade'].dropna().cumprod().values[-1]-1
    else: 
        rolling_perc = 0
    return 100*lev*rolling_perc

@st.experimental_memo
def filt_df(df, cheader, symbol_selections):
    """
        Inputs: df (pd.DataFrame), cheader (str) and symbol_selections (list[str]).
        
        Returns a filtered pd.DataFrame containing only data that matches symbol_selections (list[str])
        from df[cheader].
    """
    
    df = df.copy()
    df = df[df[cheader].isin(symbol_selections)]

    return df

@st.experimental_memo
def my_style(v, props=''):
    props = 'color:red' if v < 0 else 'color:green'
    return props

@st.cache(ttl=24*3600, allow_output_mutation=True)
def load_data(filename, otimeheader,fmat):
    df = pd.read_csv(open(filename,'r'), sep='\t') # so as not to mutate cached value 
    df.columns = ['Trade','Signal','Entry Date','Buy Price', 'Sell Price','Exit Date', 'P/L per token', 'P/L %']
    
    df['Buy Price'] = df['Buy Price'].str.replace('$', '', regex=True)
    df['Sell Price'] = df['Sell Price'].str.replace('$', '', regex=True)
    df['Buy Price'] = df['Buy Price'].str.replace(',', '', regex=True)
    df['Sell Price'] = df['Sell Price'].str.replace(',', '', regex=True)
    df['P/L per token'] = df['P/L per token'].str.replace('$', '', regex=True)
    df['P/L per token'] = df['P/L per token'].str.replace(',', '', regex=True)
    df['P/L %'] = df['P/L %'].str.replace('%', '', regex=True)

    df['Buy Price'] = pd.to_numeric(df['Buy Price'])
    df['Sell Price'] = pd.to_numeric(df['Sell Price'])
    df['P/L per token'] = pd.to_numeric(df['P/L per token'])
    df['P/L %'] = pd.to_numeric(df['P/L %'])

    dateheader = 'Date'
    theader = 'Time'

    df[dateheader] = [tradetimes.split(" ")[0] for tradetimes in df[otimeheader].values]
    df[theader] = [tradetimes.split(" ")[1] for tradetimes in df[otimeheader].values]

    df[otimeheader]= [dateutil.parser.parse(date+' '+time)
                              for date,time in zip(df[dateheader],df[theader])]

    df[otimeheader] = pd.to_datetime(df[otimeheader])
    df['Exit Date'] = pd.to_datetime(df['Exit Date'])
    df.sort_values(by=otimeheader, inplace=True)

    df[dateheader] = [dateutil.parser.parse(date).date() for date in df[dateheader]]
    df[theader] = [dateutil.parser.parse(time).time() for time in df[theader]]
    df['Trade'] = [i+1 for i in range(len(df))] #reindex
    
    return df

def runapp():
    bot_selections = "French Toast"
    otimeheader = 'Entry Date'
    plheader = 'P/L %'
    fmat = '%Y-%m-%d %H:%M:%S'
    dollar_cap = 30000.00
    fees = .075/100
    st.header(f"{bot_selections} Performance Dashboard :bread: :moneybag:")
    st.write("Welcome to the Trading Bot Dashboard by BreadBytes! You can use this dashboard to track " +
                 "the performance of our trading bots.")
 #   st.sidebar.header("FAQ")

 #   with st.sidebar.subheader("FAQ"):
 #       st.write(Path("FAQ_README.md").read_text())
    st.subheader("Choose your settings:")
    no_errors = True
    
    data = load_data("FT-Trade-Log.csv",otimeheader,fmat)
    df = data.copy(deep=True)
    
    dateheader = 'Date'
    theader = 'Time'

    with st.form("user input", ):
        if no_errors:
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    try:
                        startdate = st.date_input("Start Date", value=pd.to_datetime(df[otimeheader]).min())
                    except:
                        st.error("Please select your exchange or upload a supported trade log file.")
                        no_errors = False 
                with col2:
                    try:
                        enddate = st.date_input("End Date", value=datetime.today())
                    except:
                        st.error("Please select your exchange or upload a supported trade log file.")
                        no_errors = False 
                #st.sidebar.subheader("Customize your Dashboard")

                if no_errors and (enddate < startdate): 
                    st.error("End Date must be later than Start date. Please try again.")
                    no_errors = False 
            with st.container(): 
                col1,col2 = st.columns(2) 
                with col2:
                    lev = st.number_input('Leverage', min_value=1, value=1, max_value= 3, step=1)
                with col1:
                    principal_balance = st.number_input('Starting Balance', min_value=0.00, value=1000.00, max_value= dollar_cap, step=.01)

        #hack way to get button centered 
        c = st.columns(9)
        with c[4]: 
            submitted = st.form_submit_button("Get Cookin'!")           
    
    if submitted and principal_balance * lev > dollar_cap:
        lev = np.floor(dollar_cap/principal_balance)
        st.error(f"WARNING: (Starting Balance)*(Leverage) exceeds the ${dollar_cap} limit. Using maximum available leverage of {lev}")
                
    if submitted and no_errors:
        df = df[(df[dateheader] >= startdate) & (df[dateheader] <= enddate)]

        if len(df) == 0:
                st.error("There are no available trades matching your selections. Please try again!")
                no_errors = False
        if no_errors:

            signal_map = {'Long': 1, 'Short':-1} # 1 for long #-1 for short

            df['Calculated Return %'] = df['Signal'].map(signal_map)*(1-fees)*((df['Sell Price']-df['Buy Price'])/df['Buy Price'] - fees) #accounts for fees on open and close of trade 

            df['Return Per Trade'] = 1+df['Calculated Return %'].values

            df['Compounded Return'] = df['Return Per Trade'].cumprod()
            df['Balance used in Trade'] = [min(dollar_cap/lev, bal*principal_balance) for bal in df['Compounded Return']]
            df['Net P/L Per Trade'] = (df['Return Per Trade']-1)*lev*df['Balance used in Trade'] 
            df['Cumulative P/L'] = df['Net P/L Per Trade'].cumsum()
            cum_pl = df.loc[df.dropna().index[-1],'Cumulative P/L'] + principal_balance

            effective_return = 100*((cum_pl - principal_balance)/principal_balance)

            st.header(f"{bot_selections} Results")
            if len(bot_selections) > 1:
                st.metric(
                    "Total Account Balance",
                    f"${cum_pl:.2f}",
                    f"{100*(cum_pl-principal_balance)/(principal_balance):.2f} %",
                )

            st.line_chart(data=df.dropna(), x='Exit Date', y='Cumulative P/L', use_container_width=True)

            df['Per Trade Return Rate'] = df['Return Per Trade']-1

            totals = pd.DataFrame([], columns = ['# of Trades', 'Wins', 'Losses', 'Win Rate', 'Profit Factor'])
            data = get_hist_info(df.dropna(), principal_balance,'Per Trade Return Rate')
            totals.loc[len(totals)] = list(i for i in data)

            totals['Cum. P/L'] = cum_pl-principal_balance
            totals['Cum. P/L (%)'] = 100*(cum_pl-principal_balance)/principal_balance
            #results_df['Avg. P/L'] = (cum_pl-principal_balance)/results_df['# of Trades'].values[0]
            #results_df['Avg. P/L (%)'] = 100*results_df['Avg. P/L'].values[0]/principal_balance

            if df.empty:
                st.error("Oops! None of the data provided matches your selection(s). Please try again.")
            else:
                #st.dataframe(totals.style.format({'# of Trades': '{:.0f}','Wins': '{:.0f}','Losses': '{:.0f}','Win Rate': '{:.2f}%','Profit Factor' : '{:.2f}', 'Avg. P/L (%)': '{:.2f}%', 'Cum. P/L (%)': '{:.2f}%', 'Cum. P/L': '{:.2f}', 'Avg. P/L': '{:.2f}'})
            #.text_gradient(subset=['Win Rate'],cmap="RdYlGn", vmin = 0, vmax = 100)\
            #.text_gradient(subset=['Profit Factor'],cmap="RdYlGn", vmin = 0, vmax = 2), use_container_width=True)
                for row in totals.itertuples():
                    col1, col2, col3, col4 = st.columns(4)
                    c1, c2, c3, c4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Total Trades",
                            f"{row._1:.0f}",
                        )
                    with c1:
                        st.metric(
                            "Profit Factor",
                            f"{row._5:.2f}",
                        )
                    with col2: 
                        st.metric(
                            "Wins",
                            f"{row.Wins:.0f}",
                        )
                    with c2:
                        st.metric(
                            "Cumulative P/L",
                            f"${row._6:.2f}",
                            f"{row._7:.2f} %",
                        )
                    with col3: 
                        st.metric(
                            "Losses",
                            f"{row.Losses:.0f}",
                        )
                    with c3:
                        st.metric(
                        "Rolling 7 Days",
                            "",#f"{(1+get_rolling_stats(df,otimeheader, 30))*principal_balance:.2f}",
                            f"{get_rolling_stats(df,lev, otimeheader, 7):.2f}%",
                        )
                        st.metric(
                        "Rolling 30 Days",
                            "",#f"{(1+get_rolling_stats(df,otimeheader, 30))*principal_balance:.2f}",
                            f"{get_rolling_stats(df,lev, otimeheader, 30):.2f}%",
                        )

                    with col4: 
                        st.metric(
                            "Win Rate",
                            f"{row._4:.1f}%",
                        )
                    with c4:
                        st.metric(
                        "Rolling 90 Days",
                            "",#f"{(1+get_rolling_stats(df,otimeheader, 30))*principal_balance:.2f}",
                            f"{get_rolling_stats(df,lev, otimeheader, 90):.2f}%",
                        )
                        st.metric(
                        "Rolling 180 Days",
                            "",#f"{(1+get_rolling_stats(df,otimeheader, 30))*principal_balance:.2f}",
                            f"{get_rolling_stats(df,lev, otimeheader, 180):.2f}%",
                        )
                        
    if submitted: 
        grouped_df = df.groupby('Exit Date').agg({'Signal':'min','Entry Date': 'min','Exit Date': 'max','Buy Price': 'mean',
                                 'Sell Price' : 'max',
                                 'P/L per token': 'mean', 
                                 'Calculated Return %' : lambda x: np.round(100*lev*x.sum(),2)})
        grouped_df.index = range(1, len(grouped_df)+1)
        grouped_df.rename(columns={'Buy Price':'Avg. Buy Price',
                                   'P/L per token':'Avg. P/L per token', 
                                   'Calculated Return %':'P/L %'}, inplace=True)        
    else: 
        grouped_df = df.groupby('Exit Date').agg({'Signal':'min','Entry Date': 'min','Exit Date': 'max','Buy Price': 'mean',
                                 'Sell Price' : 'max',
                                 'P/L per token': 'mean', 
                                 'P/L %':lambda x: np.round(x.sum()/4,2)})
        grouped_df.index = range(1, len(grouped_df)+1)
        grouped_df.rename(columns={'Buy Price':'Avg. Buy Price',
                                   'P/L per token':'Avg. P/L per token'}, inplace=True)
    st.subheader("Trade Logs")
    st.dataframe(grouped_df.style.format({'Avg. Buy Price': '${:.2f}', 'Sell Price': '${:.2f}', 'Avg. P/L per token':'${:.2f}', 'P/L %':'{:.2f}%'})\
    .applymap(my_style,subset=['Avg. P/L per token'])\
    .applymap(my_style,subset=['P/L %']), use_container_width=True)
    
if __name__ == "__main__":
    st.set_page_config(
        "Trading Bot Dashboard",
        layout="wide",
    )
    runapp()
# -


