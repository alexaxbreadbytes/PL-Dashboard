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
from  matplotlib.colors import LinearSegmentedColormap


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
        fmat = '%Y-%m-%d %H:%M:%S.%f'  
        
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

        
    if logtype == "Kraken":
        otimeheader = 'time'
        cheader = 'asset'
        plheader = 'amount'
        fmat = '%Y-%m-%d %H:%M:%S.%f'  
        
    if logtype == "OkX":
        otimeheader = '\ufeffOrder Time'
        cheader = '\ufeffInstrument'
        plheader = '\ufeffPL'
        fmat = '%Y-%m-%d %H:%M:%S'        
    
    return otimeheader.lower(), cheader.lower(), plheader.lower(), fmat
    
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
def get_rolling_stats(df, lev, otimeheader, days):
    max_roll = (df[otimeheader].max() - df[otimeheader].min()).days
    
    if max_roll >= days:
        rollend = df[otimeheader].max()-timedelta(days=days)
        rolling_df = df[df[otimeheader] >= rollend]

        if len(rolling_df) > 0:
            rolling_perc = rolling_df['Return Per Trade'].dropna().cumprod().values[-1]-1
        else: 
            rolling_perc = np.nan
    else:
        rolling_perc = np.nan
    return 100*rolling_perc
@st.experimental_memo
def cc_coding(row):
    return ['background-color: orange'] * len(row) if row['Exit Date'] <= datetime.strptime('2022-12-16 00:00:00','%Y-%m-%d %H:%M:%S').date() else [''] * len(row)
def ctt_coding(row):
    return ['background-color: orange'] * len(row) if row['Exit Date'] <= datetime.strptime('2023-01-02 00:00:00','%Y-%m-%d %H:%M:%S').date() else [''] * len(row)

@st.experimental_memo
def my_style(v, props=''):
    props = 'color:red' if v < 0 else 'color:green'
    return props

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


def load_data(filename, otimeheader, fmat):
    df = pd.read_csv(open(filename,'r'), sep='\t') # so as not to mutate cached value 
    
    if filename == "CT-Trade-Log.csv" or filename == "CC-Trade-Log.csv":
        df.columns = ['Trade','Entry Date','Buy Price', 'Sell Price','Exit Date', 'P/L per token', 'P/L %', 'Drawdown %']
        df.insert(1, 'Signal', ['Long']*len(df)) 
    else: 
        df.columns = ['Trade','Signal','Entry Date','Buy Price', 'Sell Price','Exit Date', 'P/L per token', 'P/L %']
    
    if filename != "CT-Toasted-Trade-Log.csv":
        df['Signal'] = df['Signal'].str.replace(' ', '', regex=True)
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
    df['Trade'] = df.index + 1 #reindex

    if filename == "CT-Trade-Log.csv":
        df['DCA'] = np.nan

        for exit in pd.unique(df['Exit Date']):
            df_exit = df[df['Exit Date']==exit]
            for i in range(len(df_exit)):
                ind = df_exit.index[i]
                df.loc[ind,'DCA'] = i+1
    return df


def runapp() -> None:
    st.header("Trading Bot Dashboard :bread: :moneybag:")
    st.write("Welcome to the Trading Bot Dashboard by BreadBytes! You can use this dashboard to track " +
                 "the performance of our trading bots, or upload and track your own performance data from a supported exchange.")
    
    
    if 'auth_user' not in st.session_state:
        with st.form("Login"):
            user = st.text_input("Username")
            secret = st.text_input("Password")

            submitted = st.form_submit_button("Submit")
        if submitted:
            if user == st.secrets["db_username"] and secret == st.secrets["db_password"]:
                    st.success("Success!")
                    st.session_state['auth_user'] = True
            else:
                st.success("Incorrect username and/or password. Please try again.")
                st.session_state['auth_user'] = False

    try: 
        st.session_state['auth_user'] == True
    except:
        st.error("Please log in.")
        return
    
    if st.session_state['auth_user'] == True:     
        st.sidebar.header("FAQ")

        with st.sidebar.subheader("FAQ"):
            st.markdown(Path("FAQ_README.md").read_text(), unsafe_allow_html=True)

        no_errors = True
        
        
        exchanges = ["ByBit", "BitGet", "Binance","Kraken","MEXC","OkX", "BreadBytes Historical Logs"]
        logtype = st.selectbox("Select your Exchange", options=exchanges)
        
        if logtype != "BreadBytes Historical Logs":
            uploaded_data = st.file_uploader(
            "Drag and Drop files here or click Browse files.", type=[".csv", ".xlsx"], accept_multiple_files=False
                )
            if uploaded_data is None:
                st.info("Please upload a file, or select BreadBytes Historical Logs as your exchange.")
            else:
                st.success("Your file was uploaded successfully!")

                uploadtype = uploaded_data.name.split(".")[1]
                if uploadtype == "csv":
                    df = pd.read_csv(uploaded_data)
                if uploadtype == "xlsx":
                    df = pd.read_excel(uploaded_data)

                otimeheader, cheader, plheader, fmat = get_headers(logtype)

                df.columns = [c.lower() for c in df.columns]

                if not(uploaded_data is None):
                    with st.container():
                        bot_selections = "Other"
                        if bot_selections == "Other":
                            try:
                                symbols = list(df[cheader].unique())
                                symbol_selections = st.multiselect(
                                "Select/Deselect Asset(s)", options=symbols, default=symbols
                            )
                            except:
                                st.error("Please select your exchange or upload a supported trade log file.")
                                no_errors = False
                            if no_errors and symbol_selections == None:
                                st.error("Please select at least one asset.")
                                no_errors = False


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


                if no_errors:
                    df = filt_df(df, cheader, symbol_selections)

                    if len(df) == 0:
                        st.error("There are no available trades matching your selections. Please try again!")
                        no_errors = False

                if no_errors:        
                    ## reformating / necessary calculations 
                    if logtype == 'BitGet':
                        try: 
                            badcol = df.filter(regex='Unnamed').columns.values[0] 
                        except: 
                            badcol = []
                        df = df[[col for col in df.columns if col != badcol]]
                        df = df[df[plheader] != 0]
                    if logtype == 'MEXC':
                        df = df[df[plheader] != 0]
                        # collapse on transaction ID then calculate oppsition prices!!!
                    if logtype == "Kraken":
                        df = df.replace('\r\n','', regex=True) 
                        df[otimeheader] = [str(time.split(".")[0]) for time in df[otimeheader].values]
                        df = df[df['type']=='margin']
                        df[plheader] = df[plheader]-df['fee']
                        fmat = '%Y-%m-%d %H:%M:%S'
                        if len(df) == 0:
                            st.error("File Type Error. Please upload a Ledger history file from Kraken.")
                            no_errors = False

                if no_errors:
                    dateheader = 'Trade Date'
                    theader = 'Trade Time'
                    
                    if type(df[otimeheader].values[0]) != str: #clunky fix to catch non-strings since np.datetime64 unstable
                        df[otimeheader] = [str(date) for date in df[otimeheader]]
                        
                    df[dateheader] = [tradetimes.split(" ")[0] for tradetimes in df[otimeheader].values]
                    df[theader] = [tradetimes.split(" ")[1] for tradetimes in df[otimeheader].values]

                    dfmat = fmat.split(" ")[0]
                    tfmat = fmat.split(" ")[1]

                    df[otimeheader]= [datetime.strptime(date+' '+time,fmat) 
                                              for date,time in zip(df[dateheader],df[theader])]

                    df[dateheader] = [datetime.strptime(date,dfmat).date() for date in df[dateheader].values]
                    df[theader] = [datetime.strptime(time,tfmat).time() for time in df[theader].values]

                    df[otimeheader] = pd.to_datetime(df[otimeheader])

                    df.sort_values(by=otimeheader, inplace=True)
                    df.index = range(0,len(df))

                    start = df.iloc[0][dateheader] if (not startdate) else startdate
                    stop = df.iloc[len(df)-1][dateheader] if (not enddate) else enddate

                    results_df = pd.DataFrame([], columns = ['Coin', '# of Trades', 'Wins', 'Losses', 'Win Rate', 
                                                             'Profit Factor', 'Cum. P/L', 'Cum. P/L (%)', 'Avg. P/L', 'Avg. P/L (%)'])

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
                    cmap=LinearSegmentedColormap.from_list('rg',["r", "grey", "g"], N=100) 
                    df['Cumulative P/L'] = df[plheader].cumsum()
                    if logtype == "Binance": #Binance (utc) doesnt show up in st line charts???
                        xx = dateheader
                    else: 
                        xx = otimeheader
                    st.line_chart(data=df, x=xx, y='Cumulative P/L', use_container_width=True)
                    st.subheader("Summarized Results")
                    if df.empty:
                        st.error("Oops! None of the data provided matches your selection(s). Please try again.")
                    else:
                        st.dataframe(results_df.style.format({'Win Rate': '{:.2f}%','Profit Factor' : '{:.2f}', 
                                                              'Avg. P/L (%)': '{:.2f}%', 'Cum. P/L (%)': '{:.2f}%', 
                                                              'Cum. P/L': '{:.2f}', 'Avg. P/L': '{:.2f}'})\
                    .text_gradient(subset=['Win Rate'],cmap=cmap, vmin = 0, vmax = 100)\
                    .text_gradient(subset=['Profit Factor'],cmap=cmap, vmin = 0, vmax = 2), use_container_width=True)

        if logtype == "BreadBytes Historical Logs" and no_errors:
            
            bots = ["Cinnamon Toast", "French Toast", "Short Bread", "Cosmic Cupcake"]#, "CT Toasted"]
            bot_selections = st.selectbox("Select your Trading Bot", options=bots)
            otimeheader = 'Exit Date'
            fmat = '%Y-%m-%d %H:%M:%S'
            fees = .075/100
            dollar_cap = 30000.00

            if bot_selections == "Cinnamon Toast":
                lev_cap = 5
                data = load_data("CT-Trade-Log.csv",otimeheader, fmat)
            if bot_selections == "French Toast":
                lev_cap = 3
                data = load_data("FT-Trade-Log.csv",otimeheader, fmat)
            if bot_selections == "Short Bread":
                lev_cap = 5
                data = load_data("SB-Trade-Log.csv",otimeheader, fmat)
            if bot_selections == "Cosmic Cupcake":
                lev_cap = 3
                data = load_data("CC-Trade-Log.csv",otimeheader, fmat)
            if bot_selections == "CT Toasted":
                lev_cap = 5
                data = load_data("CT-Toasted-Trade-Log.csv",otimeheader, fmat)

            df = data.copy(deep=True)

            dateheader = 'Date'
            theader = 'Time'
            
            st.subheader("Choose your settings:")
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
                            lev = st.number_input('Leverage', min_value=1, value=1, max_value= lev_cap, step=1)
                        with col1:
                            principal_balance = st.number_input('Starting Balance', min_value=0.00, value=1000.00, max_value= dollar_cap, step=.01)
                
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

                #hack way to get button centered 
                c = st.columns(9)
                with c[4]: 
                    submitted = st.form_submit_button("Get Cookin'!")           

            if submitted and principal_balance * lev > dollar_cap:
                lev = np.floor(dollar_cap/principal_balance)
                st.error(f"WARNING: (Starting Balance)*(Leverage) exceeds the ${dollar_cap} limit. Using maximum available leverage of {lev}")

            if submitted and no_errors:
                df = df[(df[dateheader] >= startdate) & (df[dateheader] <= enddate)]
                signal_map = {'Long': 1, 'Short':-1}
                
                if len(df) == 0:
                        st.error("There are no available trades matching your selections. Please try again!")
                        no_errors = False
                if no_errors:
                    if bot_selections == "Cinnamon Toast":
                        dca_map = {1: dca1/100, 2: dca2/100, 3: dca3/100, 4: dca4/100}
                        df['DCA %'] = df['DCA'].map(dca_map)
                        df['Calculated Return %'] = df['Signal'].map(signal_map)*(df['DCA %'])*(1-fees)*((df['Sell Price']-df['Buy Price'])/df['Buy Price'] - fees) #accounts for fees on open and close of trade 
                        
                        df['Return Per Trade'] = np.nan
                        df['Balance used in Trade'] = np.nan
                        df['New Balance'] = np.nan

                        
                        g = df.groupby('Exit Date').sum(numeric_only=True)['Calculated Return %'].reset_index(name='Return Per Trade')
                        df.loc[df['DCA']==1.0,'Return Per Trade'] = 1+lev*g['Return Per Trade'].values
                        
                        df['Compounded Return'] = df['Return Per Trade'].cumprod()
                        df.loc[df['DCA']==1.0,'New Balance'] = [min(dollar_cap/lev, bal*principal_balance) for bal in df.loc[df['DCA']==1.0,'Compounded Return']]
                        df.loc[df['DCA']==1.0,'Balance used in Trade'] = np.concatenate([[principal_balance], df.loc[df['DCA']==1.0,'New Balance'].values[:-1]])
                    else: 
                        df['Calculated Return %'] = df['Signal'].map(signal_map)*(1-fees)*((df['Sell Price']-df['Buy Price'])/df['Buy Price'] - fees) #accounts for fees on open and close of trade 
                        df['Return Per Trade'] = np.nan
                        g = df.groupby('Exit Date').sum(numeric_only=True)['Calculated Return %'].reset_index(name='Return Per Trade')
                        df['Return Per Trade'] = 1+lev*g['Return Per Trade'].values
                        
                        df['Compounded Return'] = df['Return Per Trade'].cumprod()
                        df['New Balance'] = [min(dollar_cap/lev, bal*principal_balance) for bal in df['Compounded Return']]
                        df['Balance used in Trade'] = np.concatenate([[principal_balance], df['New Balance'].values[:-1]])
                    df['Net P/L Per Trade'] = (df['Return Per Trade']-1)*df['Balance used in Trade']
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
                    if df.empty:
                        st.error("Oops! None of the data provided matches your selection(s). Please try again.")
                    else:
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
            
            if bot_selections == "Cinnamon Toast":
                if submitted:                     
                    grouped_df = df.groupby('Exit Date').agg({'Signal':'min','Entry Date': 'min','Exit Date': 'max','Buy Price': 'mean',
                                             'Sell Price' : 'max',
                                             'Net P/L Per Trade': 'mean', 
                                             'Calculated Return %' : lambda x: np.round(100*lev*x.sum(),2), 
                                             'DCA': 'max'})
                    grouped_df.index = range(1, len(grouped_df)+1)
                    grouped_df.rename(columns={'DCA' : '# of DCAs', 'Buy Price':'Avg. Buy Price',
                                               'Net P/L Per Trade':'Net P/L', 
                                               'Calculated Return %':'P/L %'}, inplace=True)
                else: 
                    grouped_df = df.groupby('Exit Date').agg({'Signal':'min','Entry Date': 'min','Exit Date': 'max','Buy Price': 'mean',
                                             'Sell Price' : 'max',
                                             'P/L per token': 'mean', 
                                             'P/L %':lambda x: np.round(x.sum()/4,2), 
                                             'DCA': 'max'})
                    grouped_df.index = range(1, len(grouped_df)+1)
                    grouped_df.rename(columns={'DCA' : '# of DCAs', 'Buy Price':'Avg. Buy Price',
                                               'P/L per token':'Net P/L'}, inplace=True)
                
            else: 
                if submitted: 
                    grouped_df = df.groupby('Exit Date').agg({'Signal':'min','Entry Date': 'min','Exit Date': 'max','Buy Price': 'mean',
                                             'Sell Price' : 'max',
                                             'Net P/L Per Trade': 'mean', 
                                             'Calculated Return %' : lambda x: np.round(100*lev*x.sum(),2)})
                    grouped_df.index = range(1, len(grouped_df)+1)
                    grouped_df.rename(columns={'Buy Price':'Avg. Buy Price',
                                               'Net P/L Per Trade':'Net P/L', 
                                               'Calculated Return %':'P/L %'}, inplace=True)        
                else: 
                    grouped_df = df.groupby('Exit Date').agg({'Signal':'min','Entry Date': 'min','Exit Date': 'max','Buy Price': 'mean',
                                             'Sell Price' : 'max',
                                             'P/L per token': 'mean', 
                                             'P/L %':'mean'})
                    grouped_df.index = range(1, len(grouped_df)+1)
                    grouped_df.rename(columns={'Buy Price':'Avg. Buy Price',
                                               'P/L per token':'Net P/L'}, inplace=True)
            st.subheader("Trade Logs")
            if bot_selections == "Cosmic Cupcake" or bot_selections == "CT Toasted":
                coding = cc_coding if bot_selections == "Cosmic Cupcake" else ctt_coding
                st.dataframe(grouped_df.style.format({'Avg. Buy Price': '${:.2f}', 'Sell Price': '${:.2f}', 'Net P/L':'${:.2f}', 'P/L %':'{:.2f}%'})\
                .apply(coding, axis=1)\
                .applymap(my_style,subset=['Net P/L'])\
                .applymap(my_style,subset=['P/L %']), use_container_width=True)
                new_title = '<div style="text-align: right;"><span style="background-color:orange;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> Not Live Traded</div>'
                st.markdown(new_title, unsafe_allow_html=True)
            else: 
                st.dataframe(grouped_df.style.format({'Avg. Buy Price': '${:.2f}', 'Sell Price': '${:.2f}', 'Net P/L':'${:.2f}', 'P/L %':'{:.2f}%'})\
                .applymap(my_style,subset=['Net P/L'])\
                .applymap(my_style,subset=['P/L %']), use_container_width=True)
                                
if __name__ == "__main__":
    st.set_page_config(
        "Trading Bot Dashboard",
        layout="wide",
    )
    runapp()
# -


