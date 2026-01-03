import json
import os
import plotly.express as px
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
st.set_page_config(layout="wide")

df = pd.read_csv('data/prices_all.csv')
df['date'] = pd.to_datetime(df['date'])
tickers = sorted(df['Ticker'].unique())

tab1, tab2 = st.tabs([
    "Предсказание",
    "Графики"
])
with tab1:
    
    uploaded_file = st.file_uploader(
        "Выберите CSV файл",
        type=['csv'],
    )

    selected_tickers = st.multiselect(
        "Выбери компании для анализа",
        tickers,
        default=tickers[:10] if len(tickers) > 10 else tickers
    )

    selected_dates = {}

    if selected_tickers:
        for ticker in selected_tickers:
            selected_date = st.date_input(
                f"Выбор даты для {ticker}",
                key=f"date_{ticker}"
            )
            selected_dates[ticker] = selected_date

        request_array = []

        for ticker in selected_tickers:
            request_array.append({
                "name": ticker,
                "date": selected_dates[ticker].strftime("%Y-%m")
            })

        if st.button('Запрос'):

            try:
                if len(request_array) == 1:
                    response = requests.post(
                f"{BACKEND_URL}/forward",
                    json=request_array[0],
                    timeout=30
                )
                    st.write("Делаю запрос на акции")
                    st.write(request_array[0])

                elif len(request_array) > 1:
                    st.json(request_array)
                    response = requests.post(
                    f"{BACKEND_URL}/forward_batch",
                    json=request_array,
                    timeout=30
                )
                    
                    st.write("Делаю запрос на выбранные акции:")

            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

            if response.status_code == 200:
                
                pred_data = response.json()
                st.write("Данные успешно получены!")
                if isinstance(pred_data, list):
                    for item in pred_data:
                        st.write(f'Название акции: {item.get("ticker")}')
                        st.write(f'Цена т.долларов: {item.get("y_pred")}')
                        st.markdown("---")
                elif isinstance(pred_data, dict):
                    st.write(f'Название акции: {pred_data.get("ticker")}')
                    st.write(f'Цена т.долларов: {pred_data.get("y_pred")}')
                else:
                    st.json(pred_data)
            else:
                st.error(f'Ошибка соединения ')
with tab2:
    start_date = pd.to_datetime(1/1/2008)
    end_date = datetime.now()

    st.subheader("Цены акций")
    if selected_tickers:
        filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        filtered_df = filtered_df[filtered_df['Ticker'].isin(selected_tickers)]

        if not filtered_df.empty:
            comparison_df = filtered_df.pivot(index='date', columns='Ticker', values='Close')

            fig = go.Figure()
        for ticker in selected_tickers:
            if ticker in comparison_df.columns:
                prices = comparison_df[ticker].dropna()
                if not prices.empty:
                    # Нормализация: начинаем с 100
                    normalized = (prices / prices.iloc[0]) * 100
                    fig.add_trace(go.Scatter(
                        x=normalized.index,
                        y=normalized.values,
                        mode='lines',
                        name=ticker,
                        hovertemplate=f"<b>{ticker}</b><br>" +
                                      f"Дата: %{{x|%d.%m.%Y}}<br>" +
                                      f"Цена: %{{y:.1f}}% от начальной<br>" +
                                      f"<extra></extra>",
                        line=dict(width=2)
                    ))

        fig.update_layout(
            xaxis_title="Дата",
            yaxis_title="Цена",
            height=600,
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            plot_bgcolor='white',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            showline=True,
            linewidth=1,
            linecolor='black'
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            showline=True,
            linewidth=1,
            linecolor='black'
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Нет данных для выбранных компаний в указанный период")


    #_________________________________________________________
    st.subheader("Доходность акций")

    COLORS = px.colors.qualitative.Set3

    # График 2.1: Кумулятивная доходность
    fig3 = go.Figure()

    for idx, ticker in enumerate(selected_tickers):
        if ticker in comparison_df.columns:
            prices = comparison_df[ticker].dropna()
            if len(prices) > 1:
                returns = prices.pct_change().fillna(0)
                cumulative_returns = (1 + returns).cumprod() - 1

                fig3.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns.values * 100,
                    mode='lines',
                    name=ticker,
                    hovertemplate=f"<b>{ticker}</b><br>" +
                                  f"Дата: %{{x|%d.%m.%Y}}<br>" +
                                  f"Доходность: %{{y:.1f}}%<br>" +
                                  f"<extra></extra>",
                    line=dict(width=3, color=COLORS[idx % len(COLORS)]),
                    opacity=0.8
                ))

    fig3.update_layout(
        title="Кумулятивная доходность",
        xaxis_title="Дата",
        yaxis_title="Накопленная доходность (%)",
        height=400,
        hovermode='x unified',
        template='plotly_white',
        yaxis=dict(ticksuffix="%")
    )

    st.plotly_chart(fig3, use_container_width=True)
