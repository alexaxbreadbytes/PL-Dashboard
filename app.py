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


# +
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
@st.experimental_memo    
def print_PL(amnt, thresh, extras = "" ):
    if amnt > 0:
        return color.BOLD + color.GREEN + str(amnt) + extras + color.END
    elif amnt < 0:
        return color.BOLD + color.RED + str(amnt)+ extras + color.END
    elif np.isnan(amnt):
        return str(np.nan)
    else:
        return str(amnt + extras)
    
@st.experimental_memo      
def get_headers(logtype):
    otimeheader = ""
    cheader = ""
    plheader = ""
    fmat = '%Y-%m-%d %H:%M:%S'
    
    if logtype == "ByBit":
        otimeheader = 'Create Time'
        cheader = 'Contracts'
        plheader = 'Closed P&L'
        fmat = '%Y-%m-%d %H:%M:%S'
        
    if logtype == "BitGet":
        otimeheader = 'Date'
        cheader = 'Futures'
        plheader = 'Realized P/L'
        fmat = '%Y-%m-%d %H:%M:%S'
        
    if logtype == "MEXC":
        otimeheader = 'Trade time'
        cheader = 'Futures'
        plheader = 'closing position'
        fmat = '%Y/%m/%d %H:%M'
        
    if logtype == "Binance":
        otimeheader = 'Date'
        cheader = 'Symbol'
        plheader = 'Realized Profit'
        fmat = '%Y-%m-%d %H:%M:%S'
        
    #if logtype == "Kucoin":
    #    otimeheader = 'Time'
    #    cheader = 'Contract'
    #    plheader = ''
    #    fmat = '%Y/%m/%d %H:%M:%S' 
    
    if logtype == "BreadBytes Historical Data":
        otimeheader = 'Entry Date'
        plheader = 'Calculated P/L'
        fmat = '%Y-%m-%d %H:%M:%S'
        
    if logtype == "Kraken":
        otimeheader = 'time'
        cheader = 'pair'
        plheader = 'amount'
        fmat = '%Y-%m-%d %H:%M:%S.%f'  
        
    if logtype == "OkX":
        otimeheader = '\ufeffOrder Time'
        cheader = '\ufeffInstrument'
        plheader = '\ufeffPL'
        fmat = '%Y-%m-%d %H:%M:%S'        
    
    return otimeheader, cheader, plheader, fmat
    
@st.experimental_memo    
def get_coin_info(df_coin, principal_balance,plheader):
    numtrades = int(len(df_coin))
    numwin = int(sum(df_coin[plheader] > 0))
    numloss = int(sum(df_coin[plheader] < 0))
    winrate = np.round(100*numwin/numtrades,2)
    
    grosswin = sum(df_coin[df_coin[plheader] > 0][plheader])
    grossloss = sum(df_coin[df_coin[plheader] < 0][plheader])
    if grossloss != 0:
        pfactor = -1*np.round(grosswin/grossloss,2)
    else: 
        pfactor = np.nan
    
    cum_PL = np.round(sum(df_coin[plheader].values),2)
    cum_PL_perc = np.round(100*cum_PL/principal_balance,2)
    mean_PL = np.round(sum(df_coin[plheader].values/len(df_coin)),2)
    mean_PL_perc = np.round(100*mean_PL/principal_balance,2)
    
    return numtrades, numwin, numloss, winrate, pfactor, cum_PL, cum_PL_perc, mean_PL, mean_PL_perc

@st.experimental_memo
def get_hist_info(df_coin, principal_balance,plheader):
    numtrades = int(len(df_coin))
    numwin = int(sum(df_coin[plheader] > 0))
    numloss = int(sum(df_coin[plheader] < 0))
    winrate = int(np.round(100*numwin/numtrades,2))
    
    grosswin = sum(df_coin[df_coin[plheader] > 0][plheader])
    grossloss = sum(df_coin[df_coin[plheader] < 0][plheader])
    pfactor = -1*np.round(grosswin/grossloss,2)
    return numtrades, numwin, numloss, winrate, pfactor
@st.experimental_memo
def get_rolling_stats(df,otimeheader, days):
    rollend = datetime.today()-timedelta(days=days)
    rolling_df = df[df[otimeheader] >= rollend]

    if len(rolling_df) > 0:
        rolling_perc = rolling_df['Return Per Trade'].cumprod().values[-1]-1
    else: 
        rolling_perc = 0
    return rolling_perc

@st.experimental_memo
def filt_df(
    df: pd.DataFrame, cheader : str, symbol_selections: list[str]) -> pd.DataFrame:
    """
        Inputs: df (pd.DataFrame), cheader (str) and symbol_selections (list[str]).
        
        Returns a filtered pd.DataFrame containing only data that matches symbol_selections (list[str])
        from df[cheader].
    """
    
    df = df.copy()
    df = df[df[cheader].isin(symbol_selections)]

    return df

def runapp() -> None:
    st.header("Trading Bot Dashboard :bread: :moneybag:")
    st.write("Welcome to the Trading Bot Dashboard by BreadBytes! You can use this dashboard to track " +
                 "the performance of our trading bots, or upload and track your own performance data from a supported exchange.")
    st.sidebar.header("FAQ")

    with st.sidebar.subheader("FAQ"):
        st.write(Path("FAQ_README.md").read_text())
        
    
    uploaded_data = st.file_uploader(
        "Drag and Drop files here or click Browse files.", type=[".csv", ".xlsx"], accept_multiple_files=False
    )
    
    if uploaded_data is None:
        st.info("No file uploaded. Using historical data.")
        uploaded_data = open("CT-Trade-Log.csv", "r")
    else:
        st.success("Your file was uploaded successfully!")
    
    uploadtype = uploaded_data.name.split(".")[1]
    if uploadtype == "csv":
        df = pd.read_csv(uploaded_data)
    if uploadtype == "xlsx":
        df = pd.read_excel(uploaded_data)
              
    no_errors = True

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Customize your Dashboard")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            exchanges = ["BreadBytes Historical Data","ByBit", "BitGet", "Binance","Kraken","MEXC","OkX"]
            logtype = st.selectbox("Select your Exchange", options=exchanges
                      )
            otimeheader, cheader, plheader, fmat = get_headers(logtype)
            
        with col2:
            bots = ["Cinnamon Toast", "French Toast", "Short Bread", "Other"]
            bot_selections = st.selectbox(
                "Select your Trading Bot", options=bots
                )
            
            symbol_selections = []
            if logtype != "BreadBytes Historical Data" and bot_selections == "Other":
                try:
                    symbols = list(df[cheader].unique())
                    symbol_selections = st.multiselect(
                    "Select/Deselect Asset(s)", options=symbols, default=symbols
                )
                except:
                    st.error("Please select your exchange or upload a supported trade log file.")
                    no_errors = False
                if symbol_selections == None:
                    st.error("Please select at least one asset.")
                    no_errors = False
    if logtype == "BreadBytes Historical Data": 
        if bot_selections == "Cinnamon Toast":
            uploaded_data = open("CT-Trade-Log.csv", "r")
            df = pd.read_csv(uploaded_data, sep='\t')
            
            df.columns = ['Trade','Entry Date','Buy Price', 'Sell Price','Exit Date', 'P/L per token', 'P/L %', 'Drawdown %']
            df['Signal'] = ['Long']*len(df)
        elif bot_selections == "Short Bread":
            uploaded_data = open("SB-Trade-Log.csv", "r")
            df = pd.read_csv(uploaded_data, sep='\t')
            
            df.columns = ['Trade','Signal','Entry Date','Buy Price', 'Sell Price','Exit Date', 'P/L per token', 'P/L %']
        elif bot_selections == "French Toast":
            uploaded_data = open("FT-Trade-Log.csv", "r")
            df = pd.read_csv(uploaded_data, sep='\t')
            
            df.columns = ['Trade','Signal','Entry Date','Buy Price', 'Sell Price','Exit Date', 'P/L per token', 'P/L %']
            df.dropna(inplace=True)
        else:
            st.error("Please select a Trading Bot.")
            no_errors = False

        if bot_selections == "Cinnamon Toast":
            with st.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1: 
                    dca1 = st.number_input('DCA 1 Allocation', min_value=0, value=25, max_value= 100, step=1)
                with col2: 
                    dca2 = st.number_input('DCA 2 Allocation', min_value=0, value=25, max_value= 100, step=1)
                with col3: 
                    dca3 = st.number_input('DCA 3 Allocation', min_value=0, value=25, max_value= 100, step=1)
                with col4: 
                    dca4 = st.number_input('DCA 4 Allocation', min_value=0, value=25, max_value= 100, step=1)
        else: 
            dca1=100; dca2=0; dca3=0; dca4=0;
    if no_errors:
        if logtype == 'Binance':
            otimeheader = df.filter(regex=otimeheader).columns.values[0]
            fmat = '%Y-%m-%d %H:%M:%S'
            df = df[df[plheader] != 0]
        #if logtype == "Kucoin":
        #        df = df.replace('\r\n','', regex=True) 
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
                    enddate = st.date_input("End Date", value=pd.to_datetime(df[otimeheader]).max())
                except:
                    st.error("Please select your exchange or upload a supported trade log file.")
                    no_errors = False 
            #st.sidebar.subheader("Customize your Dashboard")

            if no_errors and (enddate < startdate): 
                st.error("End Date must be later than Start date. Please try again.")
                no_errors = False 
        with st.container(): 
            col1,col2 = st.columns(2) 
            with col1:
                principal_balance = st.number_input('Starting Balance', min_value=0.00, value=1000.00, max_value= 1000000.00, step=10.00)
    
    with st.expander("Raw Trade Log"):
        st.write(df)
    
    if logtype == "BreadBytes Historical Data" and no_errors:
        #calculate historical data cumulative performance 
        df['Buy Price'] = df['Buy Price'].str.replace('$', '', regex=True)
        df['Sell Price'] = df['Sell Price'].str.replace('$', '', regex=True)
        df['Buy Price'] = df['Buy Price'].str.replace(',', '', regex=True)
        df['Sell Price'] = df['Sell Price'].str.replace(',', '', regex=True)

        df['Buy Price'] = pd.to_numeric(df['Buy Price'])
        df['Sell Price'] = pd.to_numeric(df['Sell Price'])
        
        dateheader = 'Date'
        theader = 'Time'

        df[dateheader] = [tradetimes.split(" ")[0] for tradetimes in df[otimeheader].values]
        df[theader] = [tradetimes.split(" ")[1] for tradetimes in df[otimeheader].values]

        df[otimeheader]= [dateutil.parser.parse(date+' '+time)
                                  for date,time in zip(df[dateheader],df[theader])]

        #df[dateheader] = pd.to_datetime(df[dateheader])

        df[otimeheader] = pd.to_datetime(df[otimeheader])
        df['Exit Date'] = pd.to_datetime(df['Exit Date'])
        df.sort_values(by=otimeheader, inplace=True)

        df[dateheader] = [dateutil.parser.parse(date).date() for date in df[dateheader]]
        df[theader] = [dateutil.parser.parse(time).time() for time in df[theader]]

        df = df[(df[dateheader] >= startdate) & (df[dateheader] <= enddate)]
        
        if len(df) == 0:
                st.error("There are no available trades matching your selections. Please try again!")
                no_errors = False
        if no_errors:

            df['DCA'] = np.nan

            for exit in pd.unique(df['Exit Date']):
                df_exit = df[df['Exit Date']==exit]
                for i in range(len(df_exit)):
                    ind = df_exit.index[i]
                    df.loc[ind,'DCA'] = i+1

            dca_map = {1: dca1/100, 2: dca2/100, 3: dca3/100, 4: dca4/100}

            df['DCA %'] = df['DCA'].map(dca_map)

            signal_map = {'Long': 1, 'Short':-1} # 1 for long #-1 for short

            df['Calculated Return %'] = (df['Signal'].map(signal_map)*((df['Sell Price']-df['Buy Price'])/df['Buy Price'])*(df['DCA %']))

            df['Return Per Trade'] = np.nan

            g = df.groupby('Exit Date').sum(numeric_only=True)['Calculated Return %'].reset_index(name='Return Per Trade')

            df.loc[df['DCA']==1.0,'Return Per Trade'] = 1+g['Return Per Trade'].values

            df['Compounded Return'] = df['Return Per Trade'].cumprod()
            cum_pl = df.loc[df.index[-1],'Compounded Return']*principal_balance

            effective_return = df.loc[df.index[-1],'Compounded Return'] - 1

            st.subheader(f"Results for {bot_selections}")
            if len(bot_selections) > 1:
                st.metric(
                    "Total Account Balance",
                    f"${cum_pl:.2f}",
                    f"{100*(cum_pl-principal_balance)/principal_balance:.2f} %",
                )

            st.subheader(f"Historical Performance")
            df['Cumulative P/L'] = (df['Compounded Return']-1)*principal_balance
            st.line_chart(data=df.dropna(), x='Exit Date', y='Cumulative P/L', use_container_width=True)

            df['Per Trade Return Rate'] = df['Return Per Trade']-1

            totals = pd.DataFrame([], columns = ['# of Trades', 'Wins', 'Losses', 'Win Rate', 'Profit Factor'])
            data = get_hist_info(df.dropna(), principal_balance,'Per Trade Return Rate')
            totals.loc[len(totals)] = list(i for i in data)

            totals['Cum. P/L'] = cum_pl-principal_balance
            totals['Cum. P/L (%)'] = 100*(cum_pl-principal_balance)/principal_balance
            #results_df['Avg. P/L'] = (cum_pl-principal_balance)/results_df['# of Trades'].values[0]
            #results_df['Avg. P/L (%)'] = 100*results_df['Avg. P/L'].values[0]/principal_balance

            st.header(f"{bot_selections} Results")
            if df.empty:
                st.error("Oops! None of the data provided matches your selection(s). Please try again.")
            else:
                #st.dataframe(totals.style.format({'# of Trades': '{:.0f}','Wins': '{:.0f}','Losses': '{:.0f}','Win Rate': '{:.2f}%','Profit Factor' : '{:.2f}', 'Avg. P/L (%)': '{:.2f}%', 'Cum. P/L (%)': '{:.2f}%', 'Cum. P/L': '{:.2f}', 'Avg. P/L': '{:.2f}'})
            #.text_gradient(subset=['Win Rate'],cmap="RdYlGn", vmin = 0, vmax = 100)\
            #.text_gradient(subset=['Profit Factor'],cmap="RdYlGn", vmin = 0, vmax = 2), use_container_width=True)
                for row in totals.itertuples():
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Total Trades",
                            f"{row._1:.0f}",
                        )
                        st.metric(
                            "Profit Factor",
                            f"{row._5:.2f}",
                        )
                    with col2: 
                        st.metric(
                            "Wins",
                            f"{row.Wins:.0f}",
                        )
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
                        st.metric(
                        "Rolling 7 Days",
                            "",#f"{(1+get_rolling_stats(df,otimeheader, 30))*principal_balance:.2f}",
                            f"{100*get_rolling_stats(df,otimeheader, 7):.2f}%",
                        )
                        st.metric(
                        "Rolling 90 Days",
                            "",#f"{(1+get_rolling_stats(df,otimeheader, 30))*principal_balance:.2f}",
                            f"{100*get_rolling_stats(df,otimeheader, 90):.2f}%",
                        )

                    with col4: 
                        st.metric(
                            "Win Rate",
                            f"{row._4:.1f}%",
                        )
                        st.metric(
                        "Rolling 30 Days",
                            "",#f"{(1+get_rolling_stats(df,otimeheader, 30))*principal_balance:.2f}",
                            f"{100*get_rolling_stats(df,otimeheader, 30):.2f}%",
                        )
                        st.metric(
                        "Rolling 180 Days",
                            "",#f"{(1+get_rolling_stats(df,otimeheader, 30))*principal_balance:.2f}",
                            f"{100*get_rolling_stats(df,otimeheader, 180):.2f}%",
                        )
                
    if logtype != "BreadBytes Historical Data":
        if no_errors:
            ##cheader = df.columns[df.isin(symbol_selections[0]).any()]    
            if bot_selections == "Cinnamon Toast" or bot_selections == "Short Bread":
                symbol_selections.append("ETHUSDT")
            if bot_selections == "French Toast":
                symbol_selections.append("BTCUSDT")

            df = filt_df(df, cheader, symbol_selections)

            if len(df) == 0:
                st.error("There are no available trades matching your selections. Please try again!")
                no_errors = False

        if no_errors:        
            ## reformating / necessary calculations 
            if logtype == 'BitGet':
                badcol = df.filter(regex='Unnamed').columns.values[0]
                df = df[[col for col in df.columns if col != badcol]]
                df = df[df[plheader] != 0]
            if logtype == 'MEXC':
                df = df[df[plheader] != 0]
                # collapse on transaction ID then calculate oppsition prices!!!
            if logtype == "Kraken":
                df = df.replace('\r\n','', regex=True) 
                df[otimeheader] = [str(time.split(".")[0]) for time in df[otimeheader].values]
                df = df[df['type'] == 'margin']
                fmat = '%Y-%m-%d %H:%M:%S'
                if len(df) == 0:
                    st.error("File Type Error. Please upload a Ledger history file from Kraken.")
                    no_errors = False
        if no_errors:

            dateheader = 'Trade Date'
            theader = 'Trade Time'

            df[dateheader] = [tradetimes.split(" ")[0] for tradetimes in df[otimeheader].values]
            df[theader] = [tradetimes.split(" ")[1] for tradetimes in df[otimeheader].values]

            dfmat = fmat.split(" ")[0]
            tfmat = fmat.split(" ")[1]

            df[otimeheader]= [datetime.strptime(date+' '+time,fmat) 
                                      for date,time in zip(df[dateheader],df[theader])]

            df[dateheader] = [datetime.strptime(date,dfmat).date() for date in df[dateheader].values]
            df[theader] = [datetime.strptime(time,tfmat).time() for time in df[theader].values]

            df[otimeheader] = pd.to_datetime(df[otimeheader])

            if logtype == "BreadBytes Historical Data":
                df['Exit Date'] = pd.to_datetime(df['Exit Date'])
            df.sort_values(by=otimeheader, inplace=True)
            df.index = range(0,len(df))

            start = df.iloc[0][dateheader] if (not startdate) else startdate
            stop = df.iloc[len(df)-1][dateheader] if (not enddate) else enddate

            results_df = pd.DataFrame([], columns = ['Coin', '# of Trades', 'Wins', 'Losses', 'Win Rate', 'Profit Factor', 'Cum. P/L', 'Cum. P/L (%)', 'Avg. P/L', 'Avg. P/L (%)'])

            for currency in pd.unique(df[cheader]): 
                df_coin = df[(df[cheader] == currency) & (df[dateheader] >= start) & (df[dateheader] <= stop)]
                data = get_coin_info(df_coin, principal_balance, plheader)
                results_df.loc[len(results_df)] = list([currency]) + list(i for i in data)

            if bot_selections == "Other" and len(pd.unique(df[cheader])) > 1: 
                df_dates = df[(df[dateheader] >= start) & (df[dateheader] <= stop)]
                data = get_coin_info(df_dates, principal_balance, plheader)
                results_df.loc[len(results_df)] = list(['Total']) + list(i for i in data)

            account_plural = "s" if len(bot_selections) > 1 else ""
            st.subheader(f"Results for your Account{account_plural}")
            totals = results_df[~(results_df['Coin'] == 'Total')].groupby('Coin', as_index=False).sum()
            if len(bot_selections) > 1:
                st.metric(
                    "Gains for All Accounts",
                    f"${totals['Cum. P/L'].sum():.2f}",
                    f"{totals['Cum. P/L (%)'].sum():.2f} %",
                )

            max_col = 4
            tot_rows = int(np.ceil(len(totals)/max_col))

            for r in np.arange(0,tot_rows): 
                #for column, row in zip(st.columns(len(totals)), totals.itertuples()):
                for column, row in zip(st.columns(max_col), totals.iloc[r*max_col:(r+1)*max_col].itertuples()):
                    column.metric(
                        row.Coin,
                        f"${row._7:.2f}",
                        f"{row._8:.2f} %",
                    )
            st.subheader(f"Historical Performance")

            ########### multi chart --- gross!!!! 
    #         plotdf = pd.DataFrame([], columns=[dateheader, cheader, 'Cumulative P/L'])
    #         for coin in pd.unique(symbol_selections):
    #             coin_df = df.loc[df[cheader]==coin, [dateheader, cheader,plheader]]
    #             coin_df['Cumulative P/L'] = coin_df[plheader].cumsum() #+ principal_balance
    #             coin_df[dateheader] = [str(i) for i in coin_df[dateheader]]
    #             plotdf = pd.concat([plotdf, coin_df.loc[:,[dateheader, cheader, 'Cumulative P/L']]])

    #         chart = alt.Chart(plotdf).mark_line().encode(
    #           x=alt.X(f'{dateheader}:N'),
    #           y=alt.Y('Cumulative P/L:Q'),
    #           color=alt.Color(f'{cheader}:N')
    #         )##.properties(title="Testing")
    #         st.altair_chart(chart, use_container_width=True)
            ########################

            #new_row = pd.DataFrame(dict(zip(df.columns, [None]*len(df.columns))), index = [0])
            #new_row[plheader] = principal_balance
            #new_row[dateheader] = df.loc[0,dateheader]
            #plotdf = pd.concat([new_row, df])
            #plotdf = df.melt(dateheader, var_name=cheader, value_name=plheader)
            df['Cumulative P/L'] = df[plheader].cumsum()
            st.line_chart(data=df, x=otimeheader, y='Cumulative P/L', use_container_width=True)
            #fig, ax = plt.subplots()
            #ax.plot(plotdf[dateheader],plotdf['Cumulative P/L'])
            #plt.xlims=([principal_balance, principal_balance+plotdf[plheader].max() + 10])
            #plt.show()
            #st.pyplot(fig)
            st.subheader("Summarized Results")
            if df.empty:
                st.error("Oops! None of the data provided matches your selection(s). Please try again.")
            else:
                #st.dataframe(results_df.style.background_gradient(subset=['Win Rate', 'Profit Factor', 'Cum. P/L (%)', 'Avg. P/L (%)'], cmap="RdYlGn"), width = 100)
                #st.dataframe(results_df.style.background_gradient(subset=['Profit Factor'], cmap="RdYlGn", vmin = -1, vmax=1), use_container_width=True)
                st.dataframe(results_df.style.format({'Win Rate': '{:.2f}%','Profit Factor' : '{:.2f}', 'Avg. P/L (%)': '{:.2f}%', 'Cum. P/L (%)': '{:.2f}%', 'Cum. P/L': '{:.2f}', 'Avg. P/L': '{:.2f}'})\
            .text_gradient(subset=['Win Rate'],cmap="RdYlGn", vmin = 0, vmax = 100)\
            .text_gradient(subset=['Profit Factor'],cmap="RdYlGn", vmin = 0, vmax = 2), use_container_width=True)
            #.highlight_min(subset=['Cum. P/L (%)', 'Avg. P/L (%)'], color='lightred')\
            #.highlight_max(subset=['Cum. P/L (%)', 'Avg. P/L (%)'], color='green'), use_container_width=True)
            #.background_gradient(subset=['Cum. P/L (%)', 'Avg. P/L (%)'],cmap="RdYlGn", vmin = -1, vmax = 1), use_container_width=True)
if __name__ == "__main__":
    st.set_page_config(
        "Trading Bot Dashboard",
        layout="wide",
    )
    runapp()
# -


