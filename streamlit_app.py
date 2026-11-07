"""
Santinvest — app Streamlit
Dados: Yahoo Finance via yfinance (sem chave de API).
Ferramenta de apoio à pesquisa. Não é recomendação de compra ou venda.
"""

import datetime as dt
import html
import json
import math
import re
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from urllib.parse import quote, quote_plus

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="Santinvest", page_icon="📈", layout="wide")


# ---------------------------------------------------------------- formatação
def br(v, casas=2):
    """Número no padrão brasileiro (1.234,56)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    s = f"{v:,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(v, casas=1):
    return "—" if v is None else f"{br(v, casas)}%"


def grande(v):
    """Valores grandes em mi / bi / tri."""
    if v is None:
        return "—"
    for lim, suf in ((1e12, "tri"), (1e9, "bi"), (1e6, "mi")):
        if abs(v) >= lim:
            return f"{br(v / lim, 2)} {suf}"
    return br(v, 0)


def num(info, chave):
    v = info.get(chave)
    if isinstance(v, (int, float)) and not np.isnan(v):
        return float(v)
    return None


# -------------------------------------------------------------------- dados
TIPOS_OK = {"EQUITY": "Ação", "ETF": "ETF/FII", "CRYPTOCURRENCY": "Cripto", "INDEX": "Índice"}


@st.cache_data(ttl=600, show_spinner=False)
def buscar_ativos(consulta):
    """Procura por nome ou código (ações, FIIs/ETFs, criptomoedas)."""
    consulta = consulta.strip()
    if not consulta:
        return []
    resultados = []
    try:
        quotes = yf.Search(consulta, max_results=10, news_count=0).quotes
    except Exception:
        quotes = []
    for q in quotes:
        simbolo, tipo = q.get("symbol"), q.get("quoteType")
        if not simbolo or tipo not in TIPOS_OK:
            continue
        nome = q.get("longname") or q.get("shortname") or simbolo
        bolsa = q.get("exchDisp") or q.get("exchange") or ""
        resultados.append({
            "simbolo": simbolo, "tipo": tipo, "nome": nome,
            "rotulo": f"{nome} ({simbolo}) · {TIPOS_OK[tipo]} · {bolsa}"})
    if resultados:
        return resultados

    # Plano B: tenta o texto como código (direto, B3 ou cripto)
    base = consulta.upper().replace(" ", "")
    for cand in (base, base + ".SA", base + "-USD"):
        try:
            h = yf.Ticker(cand).history(period="5d")
        except Exception:
            continue
        if not h.empty:
            tipo = "CRYPTOCURRENCY" if cand.endswith("-USD") else "EQUITY"
            resultados.append({"simbolo": cand, "tipo": tipo, "nome": cand,
                               "rotulo": f"{cand} · {TIPOS_OK[tipo]}"})
            break
    return resultados


@st.cache_data(ttl=900, show_spinner=False)
def carregar_historico(ticker):
    df = yf.Ticker(ticker).history(period="5y", auto_adjust=True)
    if df.empty:
        return df
    df.index = df.index.tz_localize(None)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def carregar_info(ticker):
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def carregar_dividendos(ticker):
    try:
        d = yf.Ticker(ticker).dividends
        d.index = d.index.tz_localize(None)
        return d
    except Exception:
        return pd.Series(dtype=float)


# -------------------------------------------------------------- indicadores
def calcular_indicadores(df):
    d = df.copy()
    c = d["Close"]

    for n in (20, 50, 200):
        d[f"SMA{n}"] = c.rolling(n).mean()

    # Bandas de Bollinger (20, 2)
    media = c.rolling(20).mean()
    desvio = c.rolling(20).std()
    d["BB_sup"] = media + 2 * desvio
    d["BB_inf"] = media - 2 * desvio

    # RSI (14, suavização de Wilder)
    delta = c.diff()
    ganho = delta.clip(lower=0)
    perda = -delta.clip(upper=0)
    mg = ganho.ewm(alpha=1 / 14, adjust=False).mean()
    mp = perda.ewm(alpha=1 / 14, adjust=False).mean()
    rs = mg / mp
    d["RSI"] = 100 - 100 / (1 + rs)

    # MACD (12, 26, 9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_sinal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_sinal"]

    # Volume
    d["VOL_MM20"] = d["Volume"].rolling(20).mean()
    return d


def analisar_tendencia(d):
    """Lê a tendência atual a partir das médias móveis."""
    u = d.iloc[-1]
    preco = u["Close"]
    itens = []

    def rotulo(media, nome):
        if pd.isna(media):
            return ("—", f"Dados insuficientes para {nome}")
        if preco > media:
            return ("Alta", f"Preço acima da {nome} ({pct((preco / media - 1) * 100)})")
        return ("Baixa", f"Preço abaixo da {nome} ({pct((preco / media - 1) * 100)})")

    curto = rotulo(u["SMA20"], "média de 20 dias")
    medio = rotulo(u["SMA50"], "média de 50 dias")
    longo = rotulo(u["SMA200"], "média de 200 dias")

    cruz = "—"
    if pd.notna(u["SMA50"]) and pd.notna(u["SMA200"]):
        cruz = (
            "SMA50 acima da SMA200 (configuração de alta)"
            if u["SMA50"] > u["SMA200"]
            else "SMA50 abaixo da SMA200 (configuração de baixa)"
        )

    inclinacao = None
    s50 = d["SMA50"].dropna()
    if len(s50) > 21:
        inclinacao = (s50.iloc[-1] / s50.iloc[-21] - 1) * 100

    itens.append(("Curto prazo", *curto))
    itens.append(("Médio prazo", *medio))
    itens.append(("Longo prazo", *longo))

    altas = sum(1 for _, r, _ in itens if r == "Alta")
    baixas = sum(1 for _, r, _ in itens if r == "Baixa")
    if altas == 3:
        geral = "Tendência de ALTA"
    elif baixas == 3:
        geral = "Tendência de BAIXA"
    else:
        geral = "Tendência LATERAL / em transição"

    return {"geral": geral, "itens": itens, "cruzamento": cruz, "inclinacao": inclinacao}


def avaliar_momento(d):
    """Pontua sinais técnicos. Quanto maior, mais favorável o momento técnico."""
    u = d.iloc[-1]
    ant = d.iloc[-2] if len(d) > 1 else u
    preco = u["Close"]
    sinais = []  # (nome, pontos, explicação)

    # RSI
    rsi = u["RSI"]
    if pd.notna(rsi):
        if rsi < 30:
            p, t = 2, f"RSI {br(rsi, 0)}: sobrevendido (possível exagero de queda)"
        elif rsi < 45:
            p, t = 1, f"RSI {br(rsi, 0)}: fraco, perto de região de compra"
        elif rsi <= 65:
            p, t = 0, f"RSI {br(rsi, 0)}: neutro"
        elif rsi <= 70:
            p, t = -1, f"RSI {br(rsi, 0)}: esticado"
        else:
            p, t = -2, f"RSI {br(rsi, 0)}: sobrecomprado (risco de correção)"
        sinais.append(("RSI (14)", p, t))

    # Bollinger
    if pd.notna(u["BB_sup"]) and u["BB_sup"] != u["BB_inf"]:
        pb = (preco - u["BB_inf"]) / (u["BB_sup"] - u["BB_inf"])
        if pb < 0.1:
            p, t = 1, "Preço junto à banda inferior de Bollinger"
        elif pb > 0.9:
            p, t = -1, "Preço junto à banda superior de Bollinger"
        else:
            p, t = 0, "Preço no meio das bandas de Bollinger"
        sinais.append(("Bollinger", p, t))

    # MACD
    if pd.notna(u["MACD_hist"]) and pd.notna(ant["MACD_hist"]):
        if u["MACD_hist"] > 0 and u["MACD_hist"] >= ant["MACD_hist"]:
            p, t = 1, "MACD acima da linha de sinal e ganhando força"
        elif u["MACD_hist"] < 0 and u["MACD_hist"] <= ant["MACD_hist"]:
            p, t = -1, "MACD abaixo da linha de sinal e perdendo força"
        else:
            p, t = 0, "MACD sem direção clara"
        sinais.append(("MACD", p, t))

    # Tendência
    if pd.notna(u["SMA200"]):
        sinais.append(
            ("Tendência longa", 1 if preco > u["SMA200"] else -1,
             "Preço acima da média de 200 dias" if preco > u["SMA200"]
             else "Preço abaixo da média de 200 dias")
        )
    if pd.notna(u["SMA50"]):
        sinais.append(
            ("Tendência média", 1 if preco > u["SMA50"] else -1,
             "Preço acima da média de 50 dias" if preco > u["SMA50"]
             else "Preço abaixo da média de 50 dias")
        )

    # Volume
    if pd.notna(u["VOL_MM20"]) and u["VOL_MM20"] > 0:
        razao = u["Volume"] / u["VOL_MM20"]
        subiu = preco >= ant["Close"]
        if razao >= 1.5 and subiu:
            p, t = 1, f"Volume {br(razao, 1)}x a média, com preço subindo (confirma alta)"
        elif razao >= 1.5 and not subiu:
            p, t = -1, f"Volume {br(razao, 1)}x a média, com preço caindo (pressão vendedora)"
        else:
            p, t = 0, f"Volume {br(razao, 1)}x a média (sem sinal forte)"
        sinais.append(("Volume", p, t))

    total = sum(p for _, p, _ in sinais)
    maximo = 7
    altas = sum(1 for _, p, _ in sinais if p > 0)
    baixas = sum(1 for _, p, _ in sinais if p < 0)
    neutros = len(sinais) - altas - baixas

    janela = d.tail(60)
    return {
        "sinais": sinais,
        "total": total,
        "maximo": maximo,
        "leitura": f"{altas} de alta · {baixas} de baixa · {neutros} neutros",
        "curto": f"{altas}↑ {baixas}↓ {neutros}→",
        "suporte": janela["Low"].min(),
        "resistencia": janela["High"].max(),
        "bb_inf": u["BB_inf"],
        "sma50": u["SMA50"],
    }


def avaliar_fundamentos(info, dy12m):
    """Checklist de referência. Os limites são genéricos e variam por setor."""
    pl = num(info, "trailingPE")
    pvp = num(info, "priceToBook")
    roe = num(info, "returnOnEquity")
    margem = num(info, "profitMargins")
    divida = num(info, "debtToEquity")
    cresc = num(info, "revenueGrowth")
    liquidez = num(info, "currentRatio")
    ev_ebitda = num(info, "enterpriseToEbitda")

    roe = roe * 100 if roe is not None else None
    margem = margem * 100 if margem is not None else None
    cresc = cresc * 100 if cresc is not None else None

    def teste(valor, cond):
        return None if valor is None else bool(cond(valor))

    criterios = [
        ("P/L", br(pl), "entre 0 e 15", teste(pl, lambda v: 0 < v <= 15)),
        ("P/VP", br(pvp), "até 2", teste(pvp, lambda v: 0 < v <= 2)),
        ("EV/EBITDA", br(ev_ebitda), "entre 0 e 10", teste(ev_ebitda, lambda v: 0 < v <= 10)),
        ("Dividend Yield (12m)", pct(dy12m), "a partir de 4%", teste(dy12m, lambda v: v >= 4)),
        ("ROE", pct(roe), "a partir de 15%", teste(roe, lambda v: v >= 15)),
        ("Margem líquida", pct(margem), "a partir de 10%", teste(margem, lambda v: v >= 10)),
        ("Dívida / Patrimônio", pct(divida), "até 100%", teste(divida, lambda v: v <= 100)),
        ("Liquidez corrente", br(liquidez), "a partir de 1", teste(liquidez, lambda v: v >= 1)),
        ("Crescimento da receita", pct(cresc), "positivo", teste(cresc, lambda v: v > 0)),
    ]
    avaliados = [c for c in criterios if c[3] is not None]
    aprovados = sum(1 for c in avaliados if c[3])
    nota = aprovados / len(avaliados) if avaliados else None
    if nota is None:
        leitura = "Sem dados suficientes"
    else:
        leitura = f"{aprovados} de {len(avaliados)} critérios de referência atendidos"
    return {
        "criterios": criterios,
        "aprovados": aprovados,
        "total": len(avaliados),
        "leitura": leitura,
    }


def dividend_yield_12m(divs, preco):
    if divs is None or divs.empty or not preco:
        return None
    corte = divs.index.max() - pd.Timedelta(days=365)
    soma = divs[divs.index > corte].sum()
    return float(soma / preco * 100) if soma > 0 else 0.0


# ----------------------------------------------------------------- gráficos
def grafico_principal(d):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=("Preço e médias", "Volume", "RSI (14)", "MACD"),
    )
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="Preço"), row=1, col=1)
    for col, nome in (("SMA20", "Média 20"), ("SMA50", "Média 50"), ("SMA200", "Média 200")):
        fig.add_trace(go.Scatter(x=d.index, y=d[col], name=nome, line=dict(width=1.3)),
                      row=1, col=1)
    for col, nome in (("BB_sup", "Bollinger sup."), ("BB_inf", "Bollinger inf.")):
        fig.add_trace(go.Scatter(x=d.index, y=d[col], name=nome,
                                 line=dict(width=1, dash="dot", color="gray")),
                      row=1, col=1)

    cores = np.where(d["Close"] >= d["Open"], "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], marker_color=cores, name="Volume"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["VOL_MM20"], name="Volume médio 20d",
                             line=dict(width=1.3, color="orange")), row=2, col=1)

    fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], name="RSI",
                             line=dict(width=1.3, color="purple")), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

    cores_h = np.where(d["MACD_hist"] >= 0, "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=d.index, y=d["MACD_hist"], marker_color=cores_h,
                         name="Histograma"), row=4, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], name="MACD",
                             line=dict(width=1.2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD_sinal"], name="Sinal",
                             line=dict(width=1.2)), row=4, col=1)

    fig.update_layout(height=950, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.05),
                      margin=dict(l=10, r=10, t=60, b=10))
    return fig


def grafico_dividendos(divs):
    por_ano = divs.groupby(divs.index.year).sum()
    fig = go.Figure(go.Bar(x=por_ano.index.astype(str), y=por_ano.values))
    fig.update_layout(title="Dividendos pagos por ano (por ação)", height=320,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ------------------------------------------------- favoritos e alertas (salvos)
ARQ_DADOS = "dados.json"
TIPOS_ALERTA = {
    "preco_abaixo": "Preço cair até",
    "preco_acima": "Preço subir até",
    "rsi_abaixo": "RSI cair até (sobrevendido)",
}


def _vazio():
    return {"favoritos": [], "alertas": [], "carteira": [], "diario": [],
            "progresso": {"aulas": [], "quiz": {}}}


def _config_gist():
    try:
        return st.secrets["GIST_TOKEN"], st.secrets["GIST_ID"]
    except Exception:
        return None, None


def _cab(token):
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}


def _ler_gist():
    token, gid = _config_gist()
    r = requests.get(f"https://api.github.com/gists/{gid}", headers=_cab(token), timeout=15)
    r.raise_for_status()
    arq = r.json().get("files", {}).get(ARQ_DADOS)
    dados = json.loads(arq.get("content") or "{}") if arq else {}
    dados.setdefault("favoritos", [])
    dados.setdefault("alertas", [])
    dados.setdefault("carteira", [])
    dados.setdefault("diario", [])
    dados.setdefault("progresso", {"aulas": [], "quiz": {}})
    return dados


def _gravar_gist(dados):
    token, gid = _config_gist()
    corpo = {"files": {ARQ_DADOS: {"content": json.dumps(dados, ensure_ascii=False, indent=2)}}}
    r = requests.patch(f"https://api.github.com/gists/{gid}", headers=_cab(token),
                       json=corpo, timeout=15)
    r.raise_for_status()


@st.cache_data(ttl=60, show_spinner=False)
def _ler_gist_cache(gid):
    return _ler_gist()


def ler_dados():
    """Devolve (dados, salvamento_ativo). Sem configuração, vale só durante a sessão."""
    token, gid = _config_gist()
    if token:
        try:
            return _ler_gist_cache(gid), True
        except Exception as e:
            st.session_state["erro_salvar"] = f"Não consegui ler o salvamento no GitHub ({e})."
    return st.session_state.setdefault("dados", _vazio()), False


def mutar(fn):
    """Lê o dado mais recente, aplica a mudança e grava."""
    token, _ = _config_gist()
    if token:
        try:
            dados = _ler_gist()
            fn(dados)
            _gravar_gist(dados)
            _ler_gist_cache.clear()
        except Exception as e:
            st.session_state["erro_salvar"] = f"Não consegui salvar no GitHub ({e})."
        return
    fn(st.session_state.setdefault("dados", _vazio()))


def alternar_favorito(escolha):
    def fn(d):
        if any(f["simbolo"] == escolha["simbolo"] for f in d["favoritos"]):
            d["favoritos"] = [f for f in d["favoritos"] if f["simbolo"] != escolha["simbolo"]]
        else:
            d["favoritos"].append({"simbolo": escolha["simbolo"], "tipo": escolha["tipo"],
                                   "nome": escolha["nome"]})
    mutar(fn)


def criar_alerta(escolha, tipo, valor):
    def fn(d):
        d["alertas"].append({
            "id": uuid.uuid4().hex[:8], "simbolo": escolha["simbolo"], "nome": escolha["nome"],
            "tipo": tipo, "valor": float(valor), "criado": dt.date.today().isoformat(),
            "disparado_em": None})
    mutar(fn)


def remover_alerta(id_):
    mutar(lambda d: d.update(alertas=[a for a in d["alertas"] if a["id"] != id_]))


def reativar_alerta(id_):
    def fn(d):
        for a in d["alertas"]:
            if a["id"] == id_:
                a["disparado_em"] = None
    mutar(fn)


def moeda_de(simbolo):
    return "R$" if simbolo.endswith(".SA") else "US$"


def alerta_atingido(a, preco, rsi):
    if a["tipo"] == "preco_abaixo":
        return preco <= a["valor"]
    if a["tipo"] == "preco_acima":
        return preco >= a["valor"]
    if a["tipo"] == "rsi_abaixo":
        return rsi is not None and rsi <= a["valor"]
    return False


@st.cache_data(ttl=900, show_spinner=False)
def resumo_ativo(ticker):
    h = carregar_historico(ticker)
    if h is None or h.empty or len(h) < 30:
        return None
    d = calcular_indicadores(h)
    u, ant = d.iloc[-1], d.iloc[-2]
    return {
        "preco": float(u["Close"]),
        "dia": (float(u["Close"]) / float(ant["Close"]) - 1) * 100,
        "rsi": float(u["RSI"]) if pd.notna(u["RSI"]) else None,
        "tendencia": analisar_tendencia(d)["geral"].replace("Tendência de ", "").replace("Tendência ", ""),
        "momento": avaliar_momento(d)["curto"],
    }


def alertas_atingidos(dados):
    saida = []
    for a in dados["alertas"]:
        if a.get("disparado_em"):
            continue
        r = resumo_ativo(a["simbolo"])
        if r and alerta_atingido(a, r["preco"], r["rsi"]):
            saida.append((a, r))
    return saida


# ----------------------------------------------------------------- notícias
def url_google_noticias(consulta, ingles=False):
    regiao = "&hl=en-US&gl=US&ceid=US:en" if ingles else "&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    return ("https://news.google.com/rss/search?q=" + quote_plus(consulta + " when:2d")
            + regiao)


@st.cache_data(ttl=300, show_spinner=False)
def ler_noticias(url, fonte_padrao):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        r.raise_for_status()
        raiz = ET.fromstring(r.content)
    except Exception:
        return []
    itens = []
    for it in raiz.iter("item"):
        titulo = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        if not titulo or not link:
            continue
        data = None
        txt = it.findtext("pubDate")
        if txt:
            try:
                data = parsedate_to_datetime(txt)
                if data.tzinfo is None:
                    data = data.replace(tzinfo=dt.timezone.utc)
            except Exception:
                data = None
        fonte = (it.findtext("source") or fonte_padrao).strip()
        itens.append({"titulo": titulo, "link": link, "data": data, "fonte": fonte})
    return itens


def ha_quanto_tempo(data):
    if data is None:
        return ""
    seg = (dt.datetime.now(dt.timezone.utc) - data).total_seconds()
    if seg < 0:
        seg = 0
    if seg < 3600:
        return f"há {max(1, int(seg // 60))} min"
    if seg < 86400:
        return f"há {int(seg // 3600)} h"
    return f"há {int(seg // 86400)} d"


def juntar_noticias(listas, limite):
    vistos, todas = set(), []
    for lista in listas:
        for n in lista:
            chave = n["titulo"].lower()[:80]
            if chave not in vistos:
                vistos.add(chave)
                todas.append(n)
    minimo = dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    todas.sort(key=lambda n: n["data"] or minimo, reverse=True)
    return todas[:limite]


def mostrar_noticias(itens):
    if not itens:
        st.info("Não consegui carregar notícias agora. Tente atualizar em instantes.")
        return
    for n in itens:
        titulo = n["titulo"].replace("[", "(").replace("]", ")")
        st.markdown(f"**[{titulo}]({n['link']})**  \n"
                    f"<small>{n['fonte']} · {ha_quanto_tempo(n['data'])}</small>",
                    unsafe_allow_html=True)


def consulta_do_ativo(escolha):
    nome = " ".join(escolha["nome"].split()[:3])
    base = escolha["simbolo"].split(".")[0].split("-")[0]
    return f"{nome} {base}"


# ------------------------------------------------------- barra de cotações
BARRA_PADRAO = [
    ("Ibovespa", "^BVSP"), ("Dólar", "USDBRL=X"), ("Euro", "EURBRL=X"),
    ("Bitcoin", "BTC-USD"), ("Ethereum", "ETH-USD"), ("PETR4", "PETR4.SA"),
    ("VALE3", "VALE3.SA"), ("ITUB4", "ITUB4.SA"), ("S&P 500", "^GSPC"),
    ("Nasdaq", "^IXIC"), ("Ouro", "GC=F"),
]


@st.cache_data(ttl=300, show_spinner=False)
def cotacoes_barra(itens):
    """itens: tupla de (rótulo, código). Busca em paralelo; ignora o que falhar."""
    def uma(item):
        rot, sim = item
        try:
            h = yf.Ticker(sim).history(period="5d")["Close"].dropna()
            if len(h) < 2:
                return None
            return rot, sim, float(h.iloc[-1]), (float(h.iloc[-1]) / float(h.iloc[-2]) - 1) * 100
        except Exception:
            return None
    with ThreadPoolExecutor(max_workers=8) as ex:
        resultados = list(ex.map(uma, itens))
    return [r for r in resultados if r]


def prefixo_moeda(simbolo):
    if simbolo.endswith(".SA") or simbolo.endswith("BRL=X"):
        return "R$ "
    if simbolo.startswith("^"):
        return ""  # índices, em pontos
    return "US$ "


def html_barra(cotacoes):
    itens = []
    for rot, sim, preco, var in cotacoes:
        cor = "#089981" if var >= 0 else "#F23645"
        seta = "▲" if var >= 0 else "▼"
        itens.append(
            f'<span class="tk-item"><b>{html.escape(rot)}</b> {prefixo_moeda(sim)}{br(preco)} '
            f'<span style="color:{cor}">{seta} {br(abs(var))}%</span></span>')
    faixa = "".join(itens)
    duracao = max(30, 5 * len(itens))
    css = (
        "<style>"
        ".tk-wrap{overflow:hidden;white-space:nowrap;background:#1E222D;"
        "border-radius:6px;padding:9px 0;margin-bottom:14px}"
        f".tk-trilho{{display:inline-block;animation:tk-rolar {duracao}s linear infinite}}"
        ".tk-trilho:hover{animation-play-state:paused}"
        ".tk-item{display:inline-block;margin:0 22px;color:#D1D4DC;font-size:14px}"
        "@keyframes tk-rolar{from{transform:translateX(0)}to{transform:translateX(-50%)}}"
        "</style>")
    return f'{css}<div class="tk-wrap"><div class="tk-trilho">{faixa}{faixa}</div></div>'


def barra_cotacoes(dados):
    itens = list(BARRA_PADRAO)
    vistos = {s for _, s in itens}
    for f in dados["favoritos"]:
        if f["simbolo"] not in vistos:
            itens.append((f["simbolo"].split(".")[0], f["simbolo"]))
            vistos.add(f["simbolo"])
    cot = cotacoes_barra(tuple(itens))
    if cot:
        st.markdown(html_barra(cot), unsafe_allow_html=True)


# ------------------------------------------------------------ notícias cripto
TEMAS_CRIPTO = {  # tema: (busca em português, busca em inglês)
    "Geral": ("criptomoedas", "cryptocurrency"),
    "Bitcoin": ("bitcoin", "bitcoin"),
    "Ethereum": ("ethereum", "ethereum"),
    "Altcoins": ("altcoins", "altcoins"),
    "Regulação": ("regulação criptomoedas", "crypto regulation"),
}
FEEDS_CRIPTO_EN = {
    "Geral": [("Cointelegraph", "https://cointelegraph.com/rss"),
              ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml")],
    "Bitcoin": [("Cointelegraph", "https://cointelegraph.com/rss/tag/bitcoin")],
    "Ethereum": [("Cointelegraph", "https://cointelegraph.com/rss/tag/ethereum")],
    "Altcoins": [("Cointelegraph", "https://cointelegraph.com/rss/tag/altcoins")],
}


def render_cripto():
    topo1, topo2 = st.columns([3, 1])
    topo1.caption("Manchetes do mercado cripto. Atualiza sozinho a cada 5 minutos.")
    if topo2.button("🔄 Atualizar", key="atualizar_cripto"):
        ler_noticias.clear()
    idioma = st.radio("Idioma", ["Português", "Inglês (fontes internacionais)"],
                      horizontal=True, key="idioma_cripto")
    tema = st.radio("Tema", list(TEMAS_CRIPTO), horizontal=True, key="tema_cripto")
    busca_pt, busca_en = TEMAS_CRIPTO[tema]

    if idioma == "Português":
        listas = [ler_noticias(url_google_noticias(busca_pt), "Google Notícias")]
    else:
        listas = [ler_noticias(url, fonte) for fonte, url in FEEDS_CRIPTO_EN.get(tema, [])]
        listas.append(ler_noticias(url_google_noticias(busca_en, ingles=True), "Google News"))
    mostrar_noticias(juntar_noticias(listas, 25))
    st.caption("Notícias de cripto misturam análise, opinião e promoção. "
               "Confira a fonte antes de tomar qualquer decisão.")


# ------------------------------------------------------------ sobre o ativo
UA = {"User-Agent": "Santinvest/1.0 (app de analise de ativos)"}


def nome_curto(nome):
    """'Petróleo Brasileiro S.A. - Petrobras' -> 'Petrobras'."""
    nome = (nome or "").strip()
    if " - " in nome:
        nome = nome.split(" - ")[-1].strip()
    for sufixo in (" USD", " S.A.", " SA", " Inc.", " Inc", " Corporation", " Corp."):
        if nome.endswith(sufixo):
            nome = nome[: -len(sufixo)].strip()
    return nome


@st.cache_data(ttl=86400, show_spinner=False)
def wiki_buscar(consulta):
    try:
        r = requests.get("https://pt.wikipedia.org/w/rest.php/v1/search/page",
                         params={"q": consulta, "limit": 4}, headers=UA, timeout=10)
        r.raise_for_status()
        return [{"titulo": p["title"], "chave": p["key"], "descricao": p.get("description") or ""}
                for p in r.json().get("pages", [])]
    except Exception:
        return []


@st.cache_data(ttl=86400, show_spinner=False)
def wiki_texto(chave):
    """Texto completo do artigo, em texto simples, com títulos de seção marcados com ==."""
    try:
        r = requests.get("https://pt.wikipedia.org/w/api.php", headers=UA, timeout=15, params={
            "action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "wiki",
            "redirects": 1, "titles": chave.replace("_", " "), "format": "json"})
        r.raise_for_status()
        paginas = r.json().get("query", {}).get("pages", {})
        for p in paginas.values():
            return p.get("extract") or ""
    except Exception:
        pass
    return ""


def separar_wiki(texto):
    """Devolve (introdução, texto da seção de história ou '')."""
    if not texto:
        return "", ""
    partes = re.split(r"(?m)^==\s*([^=].*?)\s*==\s*$", texto)
    intro = partes[0].strip()
    historia = ""
    for i in range(1, len(partes) - 1, 2):
        if partes[i].lower().startswith(("hist", "orige", "fundaç", "fundac")):
            historia = partes[i + 1].strip()
            break
    return intro, historia


def limpar_wiki(texto, limite):
    texto = re.sub(r"(?m)^=+\s*(.*?)\s*=+\s*$", r"**\1**", texto)
    if len(texto) > limite:
        corte = texto[:limite]
        ponto = corte.rfind(". ")
        texto = (corte[: ponto + 1] if ponto > limite * 0.5 else corte) + " …"
    return texto.replace("$", "\\$").replace("\n", "\n\n")


@st.cache_data(ttl=86400, show_spinner=False)
def coingecko_info(simbolo):
    base = simbolo.split("-")[0].lower()
    try:
        s = requests.get("https://api.coingecko.com/api/v3/search", params={"query": base},
                         headers=UA, timeout=10)
        s.raise_for_status()
        cand = [m for m in s.json().get("coins", []) if (m.get("symbol") or "").lower() == base]
        if not cand:
            return None
        cand.sort(key=lambda m: m.get("market_cap_rank") or 10 ** 9)
        j = requests.get(f"https://api.coingecko.com/api/v3/coins/{cand[0]['id']}", headers=UA,
                         timeout=10, params={"localization": "false", "tickers": "false",
                                             "market_data": "false", "community_data": "false",
                                             "developer_data": "false"})
        j.raise_for_status()
        j = j.json()
        site = next((u for u in (j.get("links", {}).get("homepage") or []) if u), None)
        return {
            "nome": j.get("name"), "ranking": j.get("market_cap_rank"),
            "lancamento": j.get("genesis_date"), "algoritmo": j.get("hashing_algorithm"),
            "categorias": [c for c in (j.get("categories") or []) if c][:6], "site": site,
            "descricao": (j.get("description") or {}).get("en", "")}
    except Exception:
        return None


@st.cache_data(ttl=21600, show_spinner=False)
def trajetoria(ticker):
    try:
        h = yf.Ticker(ticker).history(period="max", auto_adjust=False)
    except Exception:
        return None
    if h is None or h.empty:
        return None
    h.index = h.index.tz_localize(None)
    ajust = h["Adj Close"] if "Adj Close" in h.columns else h["Close"]
    ajust = ajust.dropna()
    ultimo_dia = ajust.index[-1]
    retornos = {}
    for anos in (1, 3, 5, 10):
        alvo = ultimo_dia - pd.DateOffset(years=anos)
        antes = ajust[ajust.index <= alvo]
        if not antes.empty:
            retornos[anos] = (float(ajust.iloc[-1]) / float(antes.iloc[-1]) - 1) * 100
    return {
        "inicio": h.index[0].date(),
        "maxima": float(h["High"].max()), "data_maxima": h["High"].idxmax().date(),
        "minima": float(h["Low"].min()), "data_minima": h["Low"].idxmin().date(),
        "retornos": retornos}


def render_sobre(escolha, info, cripto, moeda_simbolo):
    ticker = escolha["simbolo"]
    nome = info.get("longName") or info.get("shortName") or escolha["nome"]
    st.markdown(f"### Sobre {nome_curto(nome) or ticker}")

    # ---- ficha
    local = ", ".join(x for x in (info.get("city"), info.get("state"), info.get("country")) if x)
    func = info.get("fullTimeEmployees")
    ficha = [("Código", ticker), ("Tipo", {"EQUITY": "Ação", "ETF": "ETF/FII",
                                          "CRYPTOCURRENCY": "Criptomoeda", "INDEX": "Índice"}
                                         .get(info.get("quoteType") or escolha["tipo"], "—")),
             ("Setor", info.get("sector") or "—"), ("Indústria", info.get("industry") or "—"),
             ("Sede", local or "—"),
             ("Funcionários", br(float(func), 0) if isinstance(func, (int, float)) else "—"),
             ("Bolsa", info.get("fullExchangeName") or info.get("exchange") or "—"),
             ("Site", info.get("website") or "—")]
    ficha = [(k, v) for k, v in ficha if v != "—"] or ficha[:1]
    st.dataframe(pd.DataFrame(ficha, columns=["Campo", "Informação"]), hide_index=True,
                 use_container_width=True)

    # ---- cripto: CoinGecko
    if cripto:
        cg = coingecko_info(ticker)
        if cg:
            linhas = [("Nome", cg["nome"]), ("Ranking por valor de mercado", cg["ranking"]),
                      ("Data de lançamento", cg["lancamento"]), ("Algoritmo", cg["algoritmo"]),
                      ("Categorias", ", ".join(cg["categorias"]) or None), ("Site", cg["site"])]
            st.markdown("#### Sobre a criptomoeda")
            st.dataframe(pd.DataFrame([(k, str(v)) for k, v in linhas if v],
                                      columns=["Campo", "Informação"]),
                         hide_index=True, use_container_width=True)
            if cg["descricao"]:
                with st.expander("Descrição do projeto (CoinGecko, em inglês)"):
                    st.markdown(re.sub(r"<[^>]+>", "", cg["descricao"]).replace("$", "\\$")[:2500])

    # ---- Wikipédia: o que é e história
    st.markdown("#### O que é e história")
    padrao = nome_curto(nome) or ticker.split(".")[0]
    consulta = st.text_input("Buscar na Wikipédia", value=padrao, key=f"wiki_{ticker}",
                             help="Se a página não for a do ativo, ajuste o nome e tente de novo.")
    paginas = wiki_buscar(consulta) if consulta.strip() else []
    if not paginas:
        st.info("Não encontrei uma página na Wikipédia para esse nome. Tente outro termo.")
    else:
        i = st.selectbox("Página", range(len(paginas)), key=f"wikipg_{ticker}",
                         format_func=lambda k: paginas[k]["titulo"]
                         + (f" — {paginas[k]['descricao']}" if paginas[k]["descricao"] else ""))
        pg = paginas[i]
        intro, historia = separar_wiki(wiki_texto(pg["chave"]))
        if intro:
            st.markdown(limpar_wiki(intro, 1800))
        else:
            st.info("A página não tem texto de introdução.")
        if historia:
            with st.expander("História"):
                st.markdown(limpar_wiki(historia, 4000))
        st.caption(f"Fonte: [Wikipédia]({'https://pt.wikipedia.org/wiki/' + quote(pg['chave'])}) "
                   "(CC BY-SA 4.0). Confira se a página é mesmo do ativo; o texto pode estar "
                   "desatualizado ou ter imprecisões.")

    # ---- resumo oficial do Yahoo
    if info.get("longBusinessSummary"):
        with st.expander("Resumo da empresa (Yahoo Finance, em inglês)"):
            st.markdown(info["longBusinessSummary"].replace("$", "\\$"))

    # ---- trajetória de preço
    tr = trajetoria(ticker)
    if tr:
        st.markdown("#### Trajetória de preço")
        a, b, c = st.columns(3)
        a.metric("Primeira cotação no Yahoo", f"{tr['inicio']:%d/%m/%Y}")
        b.metric("Máxima histórica", f"{moeda_simbolo} {br(tr['maxima'])}",
                 f"{tr['data_maxima']:%d/%m/%Y}", delta_color="off")
        c.metric("Mínima histórica", f"{moeda_simbolo} {br(tr['minima'])}",
                 f"{tr['data_minima']:%d/%m/%Y}", delta_color="off")
        if tr["retornos"]:
            st.dataframe(pd.DataFrame(
                [{"Período": f"{a} ano{'s' if a > 1 else ''}", "Retorno": pct(v)}
                 for a, v in tr["retornos"].items()]), hide_index=True, use_container_width=True)
        st.caption("Retornos incluem dividendos reinvestidos, quando o Yahoo informa. "
                   "Máxima e mínima são preços nominais. Passado não garante o futuro.")


# ------------------------------------------------- maiores altas e baixas (B3)
# Universo: ações líquidas da B3. A lista muda com o tempo (trocas de código, saídas
# da bolsa); códigos que não existirem mais são ignorados. Edite à vontade.
UNIVERSO_B3 = [
    "ABEV3", "ALOS3", "ALPA4", "AMBP3", "ARZZ3", "ASAI3", "AURE3", "AZZA3", "B3SA3",
    "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BEEF3", "BHIA3", "BPAC11", "BRAP4", "BRAV3",
    "BRFS3", "BRKM5", "CASH3", "CEAB3", "CMIG4", "CMIN3", "COGN3", "CPFE3", "CPLE6",
    "CRFB3", "CSAN3", "CSMG3", "CSNA3", "CURY3", "CVCB3", "CXSE3", "CYRE3", "DIRR3",
    "DXCO3", "ELET3", "ELET6", "EMBR3", "ENEV3", "ENGI11", "EQTL3", "EZTC3", "FLRY3",
    "GGBR4", "GMAT3", "GOAU4", "HAPV3", "HYPE3", "IGTI11", "IRBR3", "ITSA4", "ITUB4",
    "JBSS3", "JHSF3", "KEPL3", "KLBN11", "LREN3", "LWSA3", "MEAL3", "MGLU3", "MILS3",
    "MOTV3", "MRFG3", "MRVE3", "MULT3", "NATU3", "NGRD3", "ODPV3", "ONCO3", "PCAR3",
    "PETR3", "PETR4", "PETZ3", "PLPL3", "POMO4", "PRIO3", "QUAL3", "RADL3", "RAIL3",
    "RAIZ4", "RAPT4", "RDOR3", "RECV3", "RENT3", "SANB11", "SAPR11", "SBSP3", "SEER3",
    "SIMH3", "SLCE3", "SMFT3", "SMTO3", "STBP3", "SUZB3", "TAEE11", "TEND3", "TIMS3",
    "TOTS3", "TRPL4", "TUPY3", "UGPA3", "USIM5", "VALE3", "VAMO3", "VBBR3", "VIVA3",
    "VIVT3", "VLID3", "WEGE3", "YDUQ3", "ZAMP3",
]


def _buscar_variacoes(tickers, sufixo, moeda):
    """Variação do dia de cada ação (último fechamento ou preço atual vs. fechamento anterior)."""
    col_preco, col_vol = f"Preço ({moeda})", f"Volume ({moeda} mi)"

    def uma(t):
        try:
            h = yf.Ticker(t + sufixo).history(period="5d").dropna(subset=["Close"])
            if len(h) < 2:
                return None
            u, a = h.iloc[-1], h.iloc[-2]
            return {"Código": t, col_preco: float(u["Close"]),
                    "Variação (%)": (float(u["Close"]) / float(a["Close"]) - 1) * 100,
                    col_vol: float(u["Close"]) * float(u["Volume"]) / 1e6,
                    "Data": h.index[-1].date()}
        except Exception:
            return None
    with ThreadPoolExecutor(max_workers=10) as ex:
        res = [r for r in ex.map(uma, tickers) if r]
    if not res:
        return pd.DataFrame()
    df = pd.DataFrame(res)
    return df[df["Data"] == df["Data"].max()].reset_index(drop=True)  # tira papéis sem negócio hoje


@st.cache_data(ttl=300, show_spinner=False)
def variacoes_b3(tickers):
    return _buscar_variacoes(tickers, ".SA", "R$")


def ranking_altas_baixas(df, vol_min, n=20, col_vol="Volume (R$ mi)"):
    d = df[df[col_vol] >= vol_min].drop(columns=["Data"])
    altas = d.sort_values("Variação (%)", ascending=False).head(n).reset_index(drop=True)
    baixas = d.sort_values("Variação (%)", ascending=True).head(n).reset_index(drop=True)
    for t in (altas, baixas):
        t.index = t.index + 1
    return altas, baixas, len(d)


CONFIG_B3 = {
    "Preço (R$)": st.column_config.NumberColumn(format="%.2f"),
    "Variação (%)": st.column_config.NumberColumn(format="%+.2f%%"),
    "Volume (R$ mi)": st.column_config.NumberColumn(format="%.1f"),
}


def render_b3_hoje(dados):
    topo1, topo2 = st.columns([3, 1])
    topo1.caption("Variação do dia das ações líquidas da B3. Dados do Yahoo, com atraso de alguns "
                  "minutos; atualiza a cada 5 minutos.")
    if topo2.button("🔄 Atualizar", key="atualizar_b3"):
        variacoes_b3.clear()
    vol_min = st.slider("Volume financeiro mínimo do dia (R$ milhões)", 0.0, 100.0, 2.0, 0.5,
                        help="Filtra papéis com pouca negociação, que costumam distorcer o ranking.")
    extras = [f["simbolo"][:-3] for f in dados["favoritos"] if f["simbolo"].endswith(".SA")]
    universo = tuple(dict.fromkeys(UNIVERSO_B3 + extras))
    with st.spinner("Buscando as cotações da B3..."):
        df = variacoes_b3(universo)
    if df.empty:
        st.info("Não consegui carregar as cotações agora. Tente atualizar em instantes.")
        return
    altas, baixas, n = ranking_altas_baixas(df, vol_min)
    st.caption(f"Pregão de {df['Data'].max():%d/%m/%Y} · {n} ações analisadas "
               f"(de {len(universo)} da lista).")
    st.markdown("#### 📈 20 maiores altas do dia")
    st.dataframe(altas, column_config=CONFIG_B3, use_container_width=True)
    st.markdown("#### 📉 20 maiores baixas do dia")
    st.dataframe(baixas, column_config=CONFIG_B3, use_container_width=True)
    st.caption("O universo é uma lista fixa de ações líquidas, não toda a bolsa. Para analisar um "
               "papel, busque o código na barra lateral. Não é recomendação de compra ou venda.")


# ------------------------------------------------- maiores altas e baixas (EUA)
# Universo: ações americanas líquidas (grandes empresas e algumas de médio porte mais negociadas).
# Edite à vontade; códigos que não existirem mais são ignorados.
UNIVERSO_EUA = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "BRK-B", "JPM", "V",
    "MA", "UNH", "XOM", "LLY", "JNJ", "WMT", "PG", "HD", "COST", "ORCL", "NFLX", "BAC", "CVX",
    "ABBV", "KO", "PEP", "MRK", "AMD", "ADBE", "CRM", "TMO", "ACN", "MCD", "CSCO", "LIN", "ABT",
    "WFC", "DIS", "DHR", "INTU", "TXN", "QCOM", "IBM", "GE", "CAT", "VZ", "AMGN", "NOW", "ISRG",
    "PFE", "AXP", "SPGI", "NEE", "UBER", "RTX", "HON", "LOW", "UNP", "BKNG", "GS", "PGR", "T",
    "MS", "BLK", "ELV", "TJX", "SYK", "LMT", "VRTX", "SCHW", "C", "MDT", "ADP", "PANW", "CB",
    "DE", "REGN", "MU", "BA", "PLD", "ADI", "MMC", "CI", "BMY", "SBUX", "GILD", "LRCX", "KLAC",
    "AMT", "SO", "MO", "DUK", "ZTS", "BX", "CMG", "ANET", "SNPS", "CDNS", "NKE", "PYPL", "INTC",
    "OXY", "SLB", "COP", "MPC", "USB", "PNC", "TGT", "DG", "KHC", "MDLZ", "CL", "GIS",
    "PLTR", "COIN", "HOOD", "SOFI", "RIVN", "SNAP", "ROKU", "DKNG", "AFRM", "UPST", "MSTR", "SMCI",
    "ARM", "RBLX", "DASH", "CRWD", "DDOG", "ZS", "NET", "MDB", "TTD", "ABNB", "SNOW", "SHOP",
    "PINS", "LYFT", "F", "GM", "AAL", "DAL", "UAL", "CCL", "NCLH", "MRNA", "NIO", "BABA", "PDD",
]

CONFIG_EUA = {
    "Preço (US$)": st.column_config.NumberColumn(format="%.2f"),
    "Variação (%)": st.column_config.NumberColumn(format="%+.2f%%"),
    "Volume (US$ mi)": st.column_config.NumberColumn(format="%.1f"),
}


@st.cache_data(ttl=300, show_spinner=False)
def variacoes_eua(tickers):
    return _buscar_variacoes(tickers, "", "US$")


def render_eua_hoje(dados):
    topo1, topo2 = st.columns([3, 1])
    topo1.caption("Variação do dia das ações americanas líquidas. Dados do Yahoo, com possível atraso; "
                  "atualiza a cada 5 minutos.")
    if topo2.button("🔄 Atualizar", key="atualizar_eua"):
        variacoes_eua.clear()
    vol_min = st.slider("Volume financeiro mínimo do dia (US$ milhões)", 0.0, 500.0, 20.0, 5.0,
                        key="eua_vol",
                        help="Filtra papéis com pouca negociação, que costumam distorcer o ranking.")
    extras = [f["simbolo"] for f in dados["favoritos"]
              if not f["simbolo"].endswith((".SA", "-USD", "=X", "=F")) and not f["simbolo"].startswith("^")]
    universo = tuple(dict.fromkeys(UNIVERSO_EUA + extras))
    with st.spinner("Buscando as cotações dos EUA..."):
        df = variacoes_eua(universo)
    if df.empty:
        st.info("Não consegui carregar as cotações agora. Tente atualizar em instantes.")
        return
    altas, baixas, n = ranking_altas_baixas(df, vol_min, col_vol="Volume (US$ mi)")
    st.caption(f"Pregão de {df['Data'].max():%d/%m/%Y} (horário de Nova York) · {n} ações analisadas "
               f"(de {len(universo)} da lista). Preços em dólar; não inclui pré e pós-mercado.")
    st.markdown("#### 📈 20 maiores altas do dia")
    st.dataframe(altas, column_config=CONFIG_EUA, use_container_width=True)
    st.markdown("#### 📉 20 maiores baixas do dia")
    st.dataframe(baixas, column_config=CONFIG_EUA, use_container_width=True)
    st.caption("O universo é uma lista fixa de ações líquidas, não toda a bolsa americana. Para analisar um "
               "papel, busque o código na barra lateral. Não é recomendação de compra ou venda.")


# ---------------------------------------------------------------- mercado cripto
@st.cache_data(ttl=300, show_spinner=False)
def cripto_mercado(moeda):
    try:
        r = requests.get("https://api.coingecko.com/api/v3/coins/markets", headers=UA, timeout=15,
                         params={"vs_currency": moeda, "order": "market_cap_desc", "per_page": 100,
                                 "page": 1, "sparkline": "false",
                                 "price_change_percentage": "1h,24h,7d"})
        r.raise_for_status()
        linhas = []
        for c in r.json():
            def v(chave):
                x = c.get(f"price_change_percentage_{chave}_in_currency")
                return c.get(f"price_change_percentage_{chave}") if x is None else x
            linhas.append({
                "#": c.get("market_cap_rank"), "Nome": c.get("name"),
                "Símbolo": (c.get("symbol") or "").upper(),
                f"Preço ({'R$' if moeda == 'brl' else 'US$'})": c.get("current_price"),
                "1h (%)": v("1h"), "24h (%)": v("24h"), "7d (%)": v("7d"),
                "Volume 24h (mi)": (c.get("total_volume") or 0) / 1e6,
                "Valor de mercado (bi)": (c.get("market_cap") or 0) / 1e9})
        return pd.DataFrame(linhas)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def cripto_global():
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", headers=UA, timeout=10)
        r.raise_for_status()
        d = r.json().get("data", {})
        return {"total_usd": d.get("total_market_cap", {}).get("usd"),
                "var24": d.get("market_cap_change_percentage_24h_usd"),
                "btc": d.get("market_cap_percentage", {}).get("btc"),
                "eth": d.get("market_cap_percentage", {}).get("eth")}
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def medo_ganancia():
    try:
        r = requests.get("https://api.alternative.me/fng/", headers=UA, timeout=10)
        r.raise_for_status()
        d = r.json()["data"][0]
        nomes = {"Extreme Fear": "Medo extremo", "Fear": "Medo", "Neutral": "Neutro",
                 "Greed": "Ganância", "Extreme Greed": "Ganância extrema"}
        return int(d["value"]), nomes.get(d.get("value_classification"), d.get("value_classification"))
    except Exception:
        return None


def render_mercado_cripto():
    topo1, topo2 = st.columns([3, 1])
    topo1.caption("As 100 maiores criptomoedas por valor de mercado (CoinGecko). Atualiza a cada 5 minutos.")
    if topo2.button("🔄 Atualizar", key="atualizar_mc"):
        cripto_mercado.clear(); cripto_global.clear(); medo_ganancia.clear()
    moeda = "brl" if st.radio("Moeda", ["US$ (dólar)", "R$ (real)"], horizontal=True,
                              key="moeda_cripto").startswith("R$") else "usd"
    df = cripto_mercado(moeda)
    if df.empty:
        st.info("Não consegui carregar o mercado cripto agora (a fonte gratuita limita acessos). "
                "Tente atualizar em um minuto.")
        return

    g, fg = cripto_global(), medo_ganancia()
    a, b, c = st.columns(3)
    if g and g["total_usd"]:
        a.metric("Valor de mercado total", f"US$ {br(g['total_usd'] / 1e12, 2)} tri",
                 f"{br(g['var24'])}% em 24h" if g["var24"] is not None else None)
        b.metric("Dominância", f"BTC {br(g['btc'], 1)}%", f"ETH {br(g['eth'], 1)}%", delta_color="off")
    if fg:
        c.metric("Medo e Ganância", f"{fg[0]}/100", fg[1], delta_color="off")

    preco_col = [x for x in df.columns if x.startswith("Preço")][0]
    config = {
        preco_col: st.column_config.NumberColumn(format="%.4f"),
        "1h (%)": st.column_config.NumberColumn(format="%+.2f%%"),
        "24h (%)": st.column_config.NumberColumn(format="%+.2f%%"),
        "7d (%)": st.column_config.NumberColumn(format="%+.2f%%"),
        "Volume 24h (mi)": st.column_config.NumberColumn(format="%.0f"),
        "Valor de mercado (bi)": st.column_config.NumberColumn(format="%.2f"),
    }
    com24 = df.dropna(subset=["24h (%)"])
    cols = ["Nome", "Símbolo", preco_col, "24h (%)", "7d (%)"]
    st.markdown("#### 🚀 10 maiores altas em 24h (entre as 100 maiores)")
    st.dataframe(com24.sort_values("24h (%)", ascending=False).head(10)[cols].reset_index(drop=True),
                 column_config=config, hide_index=True, use_container_width=True)
    st.markdown("#### 🔻 10 maiores baixas em 24h (entre as 100 maiores)")
    st.dataframe(com24.sort_values("24h (%)").head(10)[cols].reset_index(drop=True),
                 column_config=config, hide_index=True, use_container_width=True)
    st.markdown("#### Top 100 por valor de mercado")
    st.dataframe(df, column_config=config, hide_index=True, use_container_width=True)
    st.caption("Criptomoedas são muito voláteis. Os dados vêm de fontes gratuitas e podem atrasar. "
               "Não é recomendação de compra ou venda.")


# ------------------------------------------------------------ carteira e risco
def fx_usdbrl():
    r = resumo_ativo("USDBRL=X")
    return r["preco"] if r else None


@st.cache_data(ttl=21600, show_spinner=False)
def retorno_ibov(data_iso):
    try:
        h = yf.Ticker("^BVSP").history(start=data_iso)["Close"].dropna()
        return (float(h.iloc[-1]) / float(h.iloc[0]) - 1) * 100 if len(h) >= 2 else None
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def retorno_cdi(data_iso):
    """CDI acumulado desde a data (série 12 do Banco Central, % ao dia)."""
    try:
        ini, fim = dt.date.fromisoformat(data_iso), dt.date.today()
        if ini >= fim:
            return None
        r = requests.get("https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados", timeout=15,
                         params={"formato": "json", "dataInicial": ini.strftime("%d/%m/%Y"),
                                 "dataFinal": fim.strftime("%d/%m/%Y")})
        r.raise_for_status()
        fator = 1.0
        for x in r.json():
            fator *= 1 + float(x["valor"]) / 100
        return (fator - 1) * 100
    except Exception:
        return None


def atr_diario(simbolo):
    h = carregar_historico(simbolo)
    if h is None or h.empty or len(h) < 20:
        return None
    c = h["Close"]
    tr = pd.concat([h["High"] - h["Low"], (h["High"] - c.shift()).abs(),
                    (h["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])


def eh_cripto(simbolo, tipo=None):
    return tipo == "CRYPTOCURRENCY" or simbolo.endswith("-USD")


# ---- posições
def salvar_posicao():
    ss = st.session_state
    esc = ss.get("_escolha")
    qtd, pm = float(ss.get("pos_qtd") or 0), float(ss.get("pos_pm") or 0)
    if not esc or qtd <= 0 or pm <= 0:
        ss["erro_salvar"] = "Escolha um ativo e informe quantidade e preço médio maiores que zero."
        return
    data = ss.get("pos_data") if ss.get("pos_usa_data") else None
    pos = {"id": uuid.uuid4().hex[:8], "simbolo": esc["simbolo"], "nome": esc["nome"],
           "lado": "short" if ss.get("pos_lado") == "Short" else "long",
           "qtd": qtd, "pm": pm, "data": data.isoformat() if data else None}
    mutar(lambda d: d.setdefault("carteira", []).append(pos))


def remover_posicao():
    id_ = st.session_state.get("pos_remover")
    mutar(lambda d: d.update(carteira=[p for p in d.get("carteira", []) if p["id"] != id_]))


def montar_carteira(carteira):
    """Uma linha por posição, com resultado e exposição em R$ (dólar convertido pelo câmbio de hoje)."""
    fx = fx_usdbrl()
    linhas = []
    for p in carteira:
        r = resumo_ativo(p["simbolo"])
        m = moeda_de(p["simbolo"])
        f = 1.0 if m == "R$" else (fx or 1.0)
        sinal = 1 if p["lado"] == "long" else -1
        preco = r["preco"] if r else None
        custo = p["pm"] * p["qtd"] * f
        resultado = sinal * (preco - p["pm"]) * p["qtd"] * f if preco is not None else None
        ibov = cdi = None
        if p.get("data") and p["lado"] == "long":
            ibov, cdi = retorno_ibov(p["data"]), retorno_cdi(p["data"])
        linhas.append({
            "id": p["id"], "Ativo": p["simbolo"], "Lado": "Long" if sinal == 1 else "Short",
            "Moeda": m, "Qtd": p["qtd"], "Preço médio": p["pm"], "Preço atual": preco,
            "Custo (R$)": custo,
            "Exposição (R$)": preco * p["qtd"] * f if preco is not None else None,
            "Resultado (R$)": resultado,
            "Resultado (%)": resultado / custo * 100 if resultado is not None and custo else None,
            "Ibovespa no período (%)": ibov, "CDI no período (%)": cdi,
            "Desde": p.get("data") or "—", "Cripto": eh_cripto(p["simbolo"])})
    df = pd.DataFrame(linhas)
    if not df.empty:
        total = df["Exposição (R$)"].sum()
        df["Peso (%)"] = df["Exposição (R$)"] / total * 100 if total else None
    return df, fx


CONFIG_CARTEIRA = {
    "Qtd": st.column_config.NumberColumn(format="%.6g"),
    "Preço médio": st.column_config.NumberColumn(format="%.4f"),
    "Preço atual": st.column_config.NumberColumn(format="%.4f"),
    "Custo (R$)": st.column_config.NumberColumn(format="%.2f"),
    "Exposição (R$)": st.column_config.NumberColumn(format="%.2f"),
    "Resultado (R$)": st.column_config.NumberColumn(format="%+.2f"),
    "Resultado (%)": st.column_config.NumberColumn(format="%+.2f%%"),
    "Peso (%)": st.column_config.NumberColumn(format="%.1f%%"),
    "Ibovespa no período (%)": st.column_config.NumberColumn(format="%+.1f%%"),
    "CDI no período (%)": st.column_config.NumberColumn(format="%+.1f%%"),
}


def render_carteira(dados, escolha):
    carteira = dados.get("carteira", [])
    st.markdown("### 💼 Carteira")
    with st.expander("➕ Adicionar posição", expanded=not carteira):
        if escolha:
            st.write(f"Ativo: **{escolha['nome']}** ({escolha['simbolo']}), o escolhido na barra lateral.")
        else:
            st.info("Escolha um ativo na barra lateral para adicionar uma posição.")
        st.radio("Lado", ["Long", "Short"], horizontal=True, key="pos_lado")
        c1, c2 = st.columns(2)
        c1.number_input("Quantidade", min_value=0.0, value=0.0, step=1.0, format="%.6g", key="pos_qtd")
        c2.number_input("Preço médio de entrada", min_value=0.0, value=0.0, step=0.01,
                        format="%.4f", key="pos_pm")
        st.checkbox("Informar a data de entrada (para comparar com Ibovespa e CDI)", key="pos_usa_data")
        if st.session_state.get("pos_usa_data"):
            st.date_input("Data de entrada", value=dt.date.today(), max_value=dt.date.today(),
                          key="pos_data")
        st.button("Adicionar à carteira", on_click=salvar_posicao, type="primary", disabled=not escolha)
        st.caption("Cada compra vira uma linha. Valores em dólar (cripto, ações dos EUA) são "
                   "convertidos pelo câmbio de hoje.")

    if not carteira:
        st.info("Sua carteira está vazia. Adicione posições acima.")
        return
    df, fx = montar_carteira(carteira)
    if fx is None and (df["Moeda"] == "US$").any():
        st.warning("Não consegui o câmbio do dólar agora: valores em dólar aparecem sem conversão.")

    custo, expo = df["Custo (R$)"].sum(), df["Exposição (R$)"].sum()
    resultado = df["Resultado (R$)"].sum()
    a, b, c = st.columns(3)
    a.metric("Investido", f"R$ {br(custo)}")
    b.metric("Exposição atual", f"R$ {br(expo)}")
    c.metric("Resultado", f"R$ {br(resultado)}", f"{br(resultado / custo * 100) if custo else '0'}%")
    mostrar = df.drop(columns=["id", "Cripto"])
    st.dataframe(mostrar, column_config=CONFIG_CARTEIRA, hide_index=True, use_container_width=True)

    por_ativo = df.groupby("Ativo")["Exposição (R$)"].sum()
    if expo:
        pesos = (por_ativo / expo * 100).sort_values(ascending=False)
        st.markdown("#### Distribuição")
        st.bar_chart(pesos.rename("Peso (%)"))
        cripto_pct = df.loc[df["Cripto"], "Exposição (R$)"].sum() / expo * 100
        st.caption(f"Cripto: {br(cripto_pct, 1)}% da exposição.")
        if pesos.iloc[0] > 30:
            st.warning(f"Concentração: {pesos.index[0]} é {br(pesos.iloc[0], 0)}% da carteira. "
                       "Muitos investidores definem um teto por ativo; defina o seu.")
    if df["Ibovespa no período (%)"].notna().any():
        st.caption("Ibovespa e CDI: retorno acumulado desde a data de entrada de cada compra "
                   "(só para posições long com data).")

    st.selectbox("Remover posição", [p["id"] for p in carteira], key="pos_remover",
                 format_func=lambda i: next(
                     f"{p['simbolo']} · {p['lado']} · {br(p['qtd'], 4)} a {br(p['pm'])}"
                     for p in carteira if p["id"] == i))
    st.button("Remover", on_click=remover_posicao)
    st.caption("Não é recomendação. Resultado não inclui custos, impostos nem dividendos.")


# ---- diário de operações
def salvar_operacao():
    ss = st.session_state
    esc = ss.get("_escolha")
    ent, sai, qtd = (float(ss.get(k) or 0) for k in ("op_ent", "op_sai", "op_qtd"))
    if not esc or ent <= 0 or sai <= 0 or qtd <= 0:
        ss["erro_salvar"] = "Escolha um ativo e informe entrada, saída e quantidade maiores que zero."
        return
    stop = float(ss.get("op_stop") or 0)
    op = {"id": uuid.uuid4().hex[:8], "simbolo": esc["simbolo"], "nome": esc["nome"],
          "lado": "short" if ss.get("op_lado") == "Short" else "long", "entrada": ent,
          "saida": sai, "qtd": qtd, "stop": stop if stop > 0 else None,
          "data": (ss.get("op_data") or dt.date.today()).isoformat(),
          "nota": (ss.get("op_nota") or "").strip()[:200]}
    mutar(lambda d: d.setdefault("diario", []).append(op))


def remover_operacao():
    id_ = st.session_state.get("op_remover")
    mutar(lambda d: d.update(diario=[o for o in d.get("diario", []) if o["id"] != id_]))


def montar_diario(ops, fx):
    linhas = []
    for o in ops:
        sinal = 1 if o["lado"] == "long" else -1
        f = 1.0 if moeda_de(o["simbolo"]) == "R$" else (fx or 1.0)
        res = sinal * (o["saida"] - o["entrada"]) * o["qtd"] * f
        risco_un = (o["entrada"] - o["stop"]) * sinal if o.get("stop") else None
        r = sinal * (o["saida"] - o["entrada"]) / risco_un if risco_un and risco_un > 0 else None
        linhas.append({"id": o["id"], "Data": o["data"], "Ativo": o["simbolo"],
                       "Lado": "Long" if sinal == 1 else "Short", "Entrada": o["entrada"],
                       "Saída": o["saida"], "Qtd": o["qtd"], "Resultado (R$)": res, "R": r,
                       "Nota": o.get("nota", "")})
    return pd.DataFrame(linhas).sort_values("Data", kind="stable").reset_index(drop=True)


def estatisticas_diario(df):
    res = df["Resultado (R$)"]
    n = len(df)
    ganhos, perdas = res[res > 0], res[res < 0]
    media_g = float(ganhos.mean()) if len(ganhos) else 0.0
    media_p = float(perdas.mean()) if len(perdas) else 0.0
    curva = res.cumsum()
    com_zero = pd.concat([pd.Series([0.0]), curva], ignore_index=True)
    dd = float((com_zero.cummax() - com_zero).max())
    seq = atual = 0
    for x in res:
        atual = atual + 1 if x < 0 else 0
        seq = max(seq, atual)
    return {"n": n, "acerto": len(ganhos) / n * 100, "total": float(res.sum()),
            "media_ganho": media_g, "media_perda": media_p,
            "payoff": media_g / abs(media_p) if media_p else None,
            "esperanca": float(res.mean()),
            "esperanca_r": float(df["R"].dropna().mean()) if df["R"].notna().any() else None,
            "seq_perdas": seq, "drawdown": dd, "curva": curva}


# ---- backtest das regras (as mesmas do bot de sinais; só compra, gráfico diário)
BT_SETUPS = {
    "pullback": {"nome": "Pullback em tendência de alta", "stop_atr": 1.5, "alvos": (1.5, 3.0)},
    "rompimento": {"nome": "Rompimento com volume", "stop_atr": 2.0, "alvos": (2.0, 4.0)},
}
BT_MAX_BARRAS = 20


def bt_preparar(df):
    d = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    c, h, l = d["Close"], d["High"], d["Low"]
    d["SMA200"] = c.rolling(200).mean()
    delta = c.diff()
    mg = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    mp = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["RSI"] = 100 - 100 / (1 + mg / mp)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["VOLMM"] = d["Volume"].rolling(20).mean()
    return d


def bt_sinais(d):
    tendencia = d["Close"] > d["SMA200"]
    pullback = (tendencia & (d["RSI"].rolling(5).min() <= 40)
                & (d["Close"] > d["High"].shift(1)))
    rompimento = (tendencia & (d["Close"] > d["High"].shift(1).rolling(20).max())
                  & (d["Volume"] > 1.5 * d["VOLMM"].shift(1)))
    saida = {}
    for nome, cond in (("pullback", pullback), ("rompimento", rompimento)):
        cond = cond.fillna(False).astype(bool)
        saida[nome] = cond & ~cond.shift(1, fill_value=False)
    return saida


def bt_simular(d, indices, nome):
    """[(posição da entrada, resultado em R)]. Entrada na abertura do dia seguinte; metade sai
    no alvo 1, metade no alvo 2; se stop e alvo caem no mesmo dia, vale o stop."""
    s = BT_SETUPS[nome]
    o, h, l, c = (d[k].values for k in ("Open", "High", "Low", "Close"))
    atr = d["ATR"].values
    resultados, livre = [], 0
    for i in indices:
        if i < livre or i + 1 >= len(d) or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        entrada = o[i + 1]
        risco = s["stop_atr"] * atr[i]
        stop = entrada - risco
        t1, t2 = entrada + s["alvos"][0] * risco, entrada + s["alvos"][1] * risco
        resto, r, t1_ok = 1.0, 0.0, False
        fim = min(i + BT_MAX_BARRAS, len(d) - 1)
        saiu = fim
        for j in range(i + 1, fim + 1):
            if l[j] <= stop:
                preco = min(o[j], stop) if j > i + 1 else stop
                r += resto * (preco - entrada) / risco
                resto, saiu = 0.0, j
                break
            if not t1_ok and h[j] >= t1:
                r += 0.5 * s["alvos"][0]
                resto, t1_ok = 0.5, True
            if t1_ok and h[j] >= t2:
                r += resto * s["alvos"][1]
                resto, saiu = 0.0, j
                break
        if resto > 0:
            r += resto * (c[fim] - entrada) / risco
        resultados.append((i + 1, float(r)))
        livre = saiu + 1
    return resultados


def render_risco(dados, escolha):
    # ---------------- calculadora de posição
    st.markdown("### 🛡️ Calculadora de posição")
    if not escolha:
        st.info("Escolha um ativo na barra lateral para usar o preço atual e o ATR dele.")
    simbolo = escolha["simbolo"] if escolha else "MANUAL"
    moeda = moeda_de(simbolo) if escolha else "R$"
    r = resumo_ativo(simbolo) if escolha else None
    c1, c2 = st.columns(2)
    capital = c1.number_input("Capital total (R$)", min_value=0.0, value=10000.0, step=500.0,
                              key="rk_capital")
    risco_pct = c2.number_input("Risco por operação (% do capital)", min_value=0.1, max_value=10.0,
                                value=1.0, step=0.1, key="rk_pct")
    lado = st.radio("Lado", ["Long", "Short"], horizontal=True, key="rk_lado")
    sinal = 1 if lado == "Long" else -1
    base = round(float(r["preco"]), 4) if r else 0.0
    entrada = st.number_input(f"Preço de entrada ({moeda})", min_value=0.0, value=base,
                              step=0.01, format="%.4f", key=f"rk_ent_{simbolo}")
    atr = atr_diario(simbolo) if escolha else None
    modos = ["Informar o preço do stop"] + (["Usar ATR do ativo (gráfico diário)"] if atr else [])
    modo = st.radio("Stop", modos, key="rk_modo")
    if modo.startswith("Usar ATR"):
        k = st.number_input("Distância do stop (em ATR)", min_value=0.5, max_value=6.0, value=1.5,
                            step=0.1, key="rk_k")
        stop = entrada - sinal * k * atr
        st.caption(f"ATR(14) diário: {moeda} {br(atr, 4)} → stop em {moeda} {br(stop, 4)}")
    else:
        stop = st.number_input(f"Preço do stop ({moeda})", min_value=0.0,
                               value=round(entrada * (1 - 0.03 * sinal), 4), step=0.01,
                               format="%.4f", key=f"rk_stop_{simbolo}_{lado}")
    dist = (entrada - stop) * sinal
    if entrada <= 0 or dist <= 0:
        st.error("O stop precisa ficar abaixo da entrada (long) ou acima dela (short).")
    else:
        fx = fx_usdbrl() if moeda == "US$" else 1.0
        f = fx or 1.0
        if moeda == "US$" and fx is None:
            st.warning("Sem o câmbio do dólar agora: tratando R$ e US$ como iguais.")
        risco_brl = capital * risco_pct / 100
        bruto = (risco_brl / f) / dist
        qtd = math.floor(bruto * 1e6) / 1e6 if eh_cripto(simbolo) else math.floor(bruto)
        valor_brl = qtd * entrada * f
        perda_brl = qtd * dist * f
        a, b, c = st.columns(3)
        a.metric("Quantidade máxima", br(qtd, 6 if eh_cripto(simbolo) else 0))
        b.metric("Valor da posição", f"R$ {br(valor_brl)}",
                 f"{br(valor_brl / capital * 100, 1) if capital else '0'}% do capital", delta_color="off")
        c.metric("Perda se o stop for atingido", f"R$ {br(perda_brl)}",
                 f"{br(perda_brl / capital * 100, 2) if capital else '0'}% do capital", delta_color="off")
        if qtd <= 0:
            st.warning("Com esse risco e esse stop, não dá para comprar nem 1 unidade. "
                       "Aumente o risco, o capital ou aproxime o stop.")
        if capital and valor_brl > capital:
            st.warning("A posição é maior que o capital: exigiria alavancagem. Confira os riscos "
                       "e as regras da sua corretora.")
        elif capital and valor_brl > capital * 0.3:
            st.warning("A posição passa de 30% do capital. Concentração alta em um ativo só.")
        alvos = [{"Alvo (risco/retorno)": f"1:{br(m, 1)}",
                  f"Preço ({moeda})": entrada + sinal * m * dist,
                  "Variação (%)": sinal * m * dist / entrada * 100,
                  "Resultado (R$)": qtd * m * dist * f} for m in (1.0, 1.5, 2.0, 3.0)]
        st.dataframe(pd.DataFrame(alvos), hide_index=True, use_container_width=True,
                     column_config={f"Preço ({moeda})": st.column_config.NumberColumn(format="%.4f"),
                                    "Variação (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                                    "Resultado (R$)": st.column_config.NumberColumn(format="%+.2f")})
        st.caption("Quantidade = risco em R$ ÷ distância até o stop. Não inclui custos nem "
                   "derrapagem (preço pior que o stop). Ações da B3 podem exigir lote de 100 "
                   "(ou fracionário).")

    # ---------------- diário de operações
    st.markdown("### 📓 Diário de operações")
    ops = dados.get("diario", [])
    with st.expander("➕ Registrar operação encerrada", expanded=not ops):
        if escolha:
            st.write(f"Ativo: **{escolha['nome']}** ({escolha['simbolo']}), o escolhido na barra lateral.")
        st.radio("Lado", ["Long", "Short"], horizontal=True, key="op_lado")
        d1, d2, d3 = st.columns(3)
        d1.number_input("Entrada", min_value=0.0, value=0.0, step=0.01, format="%.4f", key="op_ent")
        d2.number_input("Saída", min_value=0.0, value=0.0, step=0.01, format="%.4f", key="op_sai")
        d3.number_input("Quantidade", min_value=0.0, value=0.0, step=1.0, format="%.6g", key="op_qtd")
        e1, e2 = st.columns(2)
        e1.number_input("Stop inicial (opcional, para calcular o R)", min_value=0.0, value=0.0,
                        step=0.01, format="%.4f", key="op_stop")
        e2.date_input("Data da saída", value=dt.date.today(), max_value=dt.date.today(), key="op_data")
        st.text_input("Nota (opcional)", max_chars=200, key="op_nota")
        st.button("Salvar operação", on_click=salvar_operacao, type="primary", disabled=not escolha)
    if not ops:
        st.caption("Nenhuma operação registrada ainda.")
    else:
        df = montar_diario(ops, fx_usdbrl())
        est = estatisticas_diario(df)
        a, b, c = st.columns(3)
        a.metric("Operações", est["n"], f"{br(est['acerto'], 0)}% positivas", delta_color="off")
        b.metric("Resultado total", f"R$ {br(est['total'])}")
        c.metric("Expectativa por operação", f"R$ {br(est['esperanca'])}",
                 f"{br(est['esperanca_r'])}R" if est["esperanca_r"] is not None else None,
                 delta_color="off")
        a2, b2, c2 = st.columns(3)
        a2.metric("Ganho médio", f"R$ {br(est['media_ganho'])}")
        b2.metric("Perda média", f"R$ {br(est['media_perda'])}")
        c2.metric("Payoff", br(est["payoff"]) if est["payoff"] is not None else "—")
        st.caption(f"Pior sequência de perdas: {est['seq_perdas']} · maior queda da curva: "
                   f"R$ {br(est['drawdown'])}")
        st.line_chart(est["curva"].rename("Resultado acumulado (R$)"))
        st.dataframe(df.drop(columns=["id"]), hide_index=True, use_container_width=True,
                     column_config={"Resultado (R$)": st.column_config.NumberColumn(format="%+.2f"),
                                    "R": st.column_config.NumberColumn(format="%+.2f")})
        st.selectbox("Remover operação", [o["id"] for o in ops], key="op_remover",
                     format_func=lambda i: next(
                         f"{o['data']} · {o['simbolo']} · {o['lado']}" for o in ops if o["id"] == i))
        st.button("Remover", key="rm_op", on_click=remover_operacao)
        st.caption("Valores em dólar usam o câmbio de hoje. Registrar tudo, inclusive os prejuízos, "
                   "é o que torna as estatísticas úteis.")

    # ---------------- backtest
    st.markdown("### 🧪 Testar as regras no histórico")
    if not escolha:
        st.info("Escolha um ativo na barra lateral.")
        return
    nome_setup = st.selectbox("Regra", list(BT_SETUPS), key="bt_regra",
                              format_func=lambda k: BT_SETUPS[k]["nome"])
    hist = carregar_historico(escolha["simbolo"])
    if hist is None or hist.empty or len(hist) < 260:
        st.info("Histórico curto demais para testar.")
        return
    d = bt_preparar(hist)
    idx = np.flatnonzero(bt_sinais(d)[nome_setup].values)
    res = bt_simular(d, idx, nome_setup)
    if not res:
        st.info("Essa regra não gerou nenhuma operação nesse ativo no período.")
    else:
        rs = pd.Series([x for _, x in res])
        a, b, c = st.columns(3)
        a.metric("Operações", len(rs), f"{br((rs > 0).mean() * 100, 0)}% positivas", delta_color="off")
        b.metric("Média por operação", f"{br(rs.mean())}R")
        c.metric("Total", f"{br(rs.sum(), 1)}R")
        s = BT_SETUPS[nome_setup]
        st.line_chart(pd.Series(rs.cumsum().values,
                                index=[d.index[i] for i, _ in res], name="Resultado acumulado (R)"))
        st.caption(
            f"Regra: {s['nome']}. Só compra, gráfico diário, ~{br(len(d) / 252, 1)} anos de dados. "
            f"Stop = {br(s['stop_atr'], 1)} x ATR; metade sai em {br(s['alvos'][0], 1)}R e metade "
            f"em {br(s['alvos'][1], 1)}R; sai no fechamento após {BT_MAX_BARRAS} pregões. "
            "1R = o risco da operação. Sem corretagem, spread nem impostos; se stop e alvo caem "
            "no mesmo dia, vale o stop. Passado não garante o futuro.")


# ------------------------------------------------------------------ comparador
# Cada critério: (chave, rótulo, pontua mais com valor "maior" ou "menor", explicação)
CRITERIOS = [
    ("pl", "P/L", "menor", "Preço dividido pelo lucro por ação: quantos anos de lucro o preço atual "
                           "'paga'. Só faz sentido com lucro positivo."),
    ("pvp", "P/VP", "menor", "Preço dividido pelo patrimônio líquido por ação."),
    ("ev_ebitda", "EV/EBITDA", "menor", "Valor da empresa (incluindo a dívida) dividido pelo resultado "
                                        "operacional antes de depreciação e amortização."),
    ("dy", "Dividend Yield 12m (%)", "maior", "Dividendos pagos nos últimos 12 meses divididos pelo preço."),
    ("roe", "ROE (%)", "maior", "Lucro dividido pelo patrimônio líquido: retorno sobre o capital dos acionistas."),
    ("margem", "Margem líquida (%)", "maior", "Quanto sobra de lucro para cada R$ 100 de receita."),
    ("divida", "Dívida/Patrimônio (%)", "menor", "Quanto a empresa deve em relação ao patrimônio. Bancos e "
                                                 "outros setores têm padrões próprios."),
    ("cresc", "Crescimento da receita (%)", "maior", "Variação da receita no período mais recente informado."),
    ("ret12", "Retorno 12 meses (%)", "maior", "Variação do preço em 12 meses. Passado não garante o futuro."),
    ("vol", "Volatilidade 12m (% a.a.)", "menor", "Oscilação anualizada dos retornos diários: quanto o preço balança."),
    ("mdd", "Maior queda 12m (%)", "maior", "Maior queda do pico ao fundo nos últimos 12 meses "
                                            "(valores menos negativos pontuam mais)."),
]
ROTULO_CRIT = {k: r for k, r, _, _ in CRITERIOS}
PRESETS_CRIT = {
    "Exemplo: valor": {"pl": 3, "pvp": 3, "ev_ebitda": 3, "divida": 1},
    "Exemplo: dividendos": {"dy": 5, "margem": 1, "divida": 2, "roe": 1},
    "Exemplo: qualidade": {"roe": 4, "margem": 3, "divida": 3, "cresc": 2},
    "Exemplo: momento": {"ret12": 4, "vol": 2, "mdd": 2},
}
MAX_COMPARAR = 8


@st.cache_data(ttl=3600, show_spinner=False)
def metricas_ativo(simbolo):
    hist = carregar_historico(simbolo)
    if hist is None or hist.empty or len(hist) < 30:
        return None
    info = carregar_info(simbolo)
    c = hist["Close"].dropna()
    cripto = eh_cripto(simbolo, info.get("quoteType"))
    ano = 365 if cripto else 252
    janela = c.tail(ano + 1)

    def pos(v):
        return v if v is not None and v > 0 else None

    ret12 = (float(janela.iloc[-1]) / float(janela.iloc[0]) - 1) * 100 if len(janela) > ano * 0.6 else None
    rets = np.log(c.tail(ano + 1)).diff().dropna()
    vol = float(rets.std() * np.sqrt(ano) * 100) if len(rets) >= 30 else None
    mdd = float((janela / janela.cummax() - 1).min() * 100) if len(janela) >= 30 else None
    roe, margem, cresc = num(info, "returnOnEquity"), num(info, "profitMargins"), num(info, "revenueGrowth")
    divida = num(info, "debtToEquity")
    dy = None
    if not cripto:
        dy = dividend_yield_12m(carregar_dividendos(simbolo), float(c.iloc[-1]))
        dy = 0.0 if dy is None else dy
    return {"pl": pos(num(info, "trailingPE")), "pvp": pos(num(info, "priceToBook")),
            "ev_ebitda": pos(num(info, "enterpriseToEbitda")), "dy": dy,
            "roe": roe * 100 if roe is not None else None,
            "margem": margem * 100 if margem is not None else None,
            "divida": divida if divida is not None and divida >= 0 else None,
            "cresc": cresc * 100 if cresc is not None else None,
            "ret12": ret12, "vol": vol, "mdd": mdd}


def pontuar(tabela, pesos):
    """tabela: linhas = ativos, colunas = chaves dos critérios (NaN = sem dado).
    Cada critério vira uma nota 0-100 pela posição do ativo entre os escolhidos (ranking, não
    magnitude, para que valores extremos não distorçam). Sem dado: o critério é ignorado para
    aquele ativo. Devolve (notas por critério, resultado com pontuação e cobertura)."""
    direcao = {k: m for k, _, m, _ in CRITERIOS}
    notas = pd.DataFrame(np.nan, index=tabela.index, columns=tabela.columns)
    for k in tabela.columns:
        validos = tabela[k].dropna()
        if validos.empty:
            continue
        if len(validos) == 1:
            notas.loc[validos.index, k] = 50.0
            continue
        pos = validos.rank(method="average", ascending=(direcao[k] == "maior"))
        notas.loc[validos.index, k] = (pos - 1) / (len(validos) - 1) * 100
    w = pd.Series({k: float(pesos.get(k, 0)) for k in tabela.columns})
    disponivel = notas.notna().mul(w).sum(axis=1)
    pontos = notas.mul(w).sum(axis=1, skipna=True) / disponivel.where(disponivel > 0)
    cobertura = disponivel / w.sum() * 100 if w.sum() > 0 else disponivel * 0
    return notas, pd.DataFrame({"Pontuação": pontos, "Cobertura (%)": cobertura})


def cmp_adicionar_busca(resultados):
    ss = st.session_state
    r = resultados[ss.get("cmp_res", 0)]
    cmp_incluir({"simbolo": r["simbolo"], "nome": r["nome"], "tipo": r["tipo"]})


def cmp_incluir(ativo):
    ss = st.session_state
    lista = ss.setdefault("cmp_lista", [])
    if not any(a["simbolo"] == ativo["simbolo"] for a in lista):
        lista.append(ativo)
    sel = list(ss.get("cmp_sel", []))
    if ativo["simbolo"] not in sel:
        ss["cmp_sel"] = sel + [ativo["simbolo"]]


def cmp_preset(nome):
    for k, _, _, _ in CRITERIOS:
        st.session_state[f"cmp_w_{k}"] = PRESETS_CRIT.get(nome, {}).get(k, 0) if nome else 0
    if nome is None:
        return


def cmp_todos_iguais():
    for k, _, _, _ in CRITERIOS:
        st.session_state[f"cmp_w_{k}"] = 1


def render_comparador(dados, escolha):
    ss = st.session_state
    st.markdown("### ⚖️ Comparador de ativos")
    st.caption("Escolha os ativos e dê o peso que quiser a cada critério. A pontuação é só o resultado "
               "dos SEUS pesos: a ferramenta não indica o que comprar.")
    if "cmp_lista" not in ss:
        ini = [{"simbolo": f["simbolo"], "nome": f["nome"], "tipo": f["tipo"]}
               for f in dados["favoritos"][:5]]
        if escolha and not any(a["simbolo"] == escolha["simbolo"] for a in ini):
            ini.append({"simbolo": escolha["simbolo"], "nome": escolha["nome"], "tipo": escolha["tipo"]})
        ss["cmp_lista"] = ini
        ss["cmp_sel"] = [a["simbolo"] for a in ini]

    # ---- escolher os ativos
    consulta = st.text_input("Adicionar ativo à comparação (nome ou código)", key="cmp_busca")
    if consulta.strip():
        achados = buscar_ativos(consulta)
        if achados:
            st.selectbox("Resultado", range(len(achados)), key="cmp_res",
                         format_func=lambda k: achados[k]["rotulo"])
            st.button("Adicionar à comparação", on_click=cmp_adicionar_busca, args=(achados,))
        else:
            st.caption("Nada encontrado para esse texto.")
    if escolha and not any(a["simbolo"] == escolha["simbolo"] for a in ss["cmp_lista"]):
        st.button(f"Adicionar {escolha['simbolo']} (da barra lateral)", on_click=cmp_incluir,
                  args=({"simbolo": escolha["simbolo"], "nome": escolha["nome"], "tipo": escolha["tipo"]},))
    nomes = {a["simbolo"]: a["nome"] for a in ss["cmp_lista"]}
    sel = st.multiselect("Ativos na comparação", list(nomes), key="cmp_sel",
                         format_func=lambda s: f"{nomes[s]} ({s})")
    if len(sel) > MAX_COMPARAR:
        st.warning(f"Comparando só os {MAX_COMPARAR} primeiros.")
        sel = sel[:MAX_COMPARAR]
    if len(sel) < 2:
        st.info("Escolha pelo menos 2 ativos para comparar.")
        return

    # ---- pesos
    with st.expander("⚙️ Pesos dos critérios (0 = ignorar, 5 = muito importante)", expanded=False):
        cols = st.columns(3)
        cols[0].button("Todos iguais", on_click=cmp_todos_iguais, key="cmp_p_iguais")
        cols[1].button("Zerar", on_click=cmp_preset, args=(None,), key="cmp_p_zero")
        for i, nome in enumerate(PRESETS_CRIT):
            cols[2].button(nome, on_click=cmp_preset, args=(nome,), key=f"cmp_p_{i}")
        for k, rot, _, _ in CRITERIOS:
            ss.setdefault(f"cmp_w_{k}", 1)
            st.slider(rot, 0, 5, key=f"cmp_w_{k}")
        st.caption("Os exemplos mostram ênfases comuns em estudos de investimento. São pontos de "
                   "partida didáticos, não recomendação.")
    pesos = {k: ss.get(f"cmp_w_{k}", 1) for k, _, _, _ in CRITERIOS}

    # ---- dados
    with st.spinner("Buscando os indicadores..."):
        linhas = {s: metricas_ativo(s) for s in sel}
    ok = {s: m for s, m in linhas.items() if m}
    for s in sel:
        if s not in ok:
            st.caption(f"Sem dados suficientes para {s}; ele ficou de fora.")
    if len(ok) < 2:
        st.info("Preciso de dados de pelo menos 2 ativos para comparar.")
        return
    chaves = [k for k, _, _, _ in CRITERIOS]
    tabela = pd.DataFrame({s: {k: (m.get(k) if m.get(k) is not None else np.nan) for k in chaves}
                           for s, m in ok.items()}).T[chaves].astype(float)
    st.markdown("#### Indicadores")
    st.dataframe(tabela.rename(columns=ROTULO_CRIT), use_container_width=True,
                 column_config={ROTULO_CRIT[k]: st.column_config.NumberColumn(format="%.2f") for k in chaves})

    if sum(pesos.values()) == 0:
        st.info("Dê peso maior que zero a pelo menos um critério para ver a pontuação.")
    else:
        notas, res = pontuar(tabela, pesos)
        res = res.sort_values("Pontuação", ascending=False)
        st.markdown("#### Pontuação pelos seus critérios")
        st.bar_chart(res["Pontuação"])
        st.dataframe(res, use_container_width=True,
                     column_config={"Pontuação": st.column_config.NumberColumn(format="%.0f"),
                                    "Cobertura (%)": st.column_config.NumberColumn(format="%.0f%%")})
        with st.expander("Notas por critério (0 a 100)"):
            usados = [k for k in chaves if pesos[k] > 0]
            st.dataframe(notas[usados].rename(columns=ROTULO_CRIT), use_container_width=True,
                         column_config={ROTULO_CRIT[k]: st.column_config.NumberColumn(format="%.0f")
                                        for k in usados})
        st.caption("Como funciona: para cada critério, o ativo recebe de 0 a 100 conforme sua posição "
                   "entre os escolhidos (com 2 ativos, uma diferença pequena vira 0 contra 100). "
                   "Dado ausente é ignorado e reduz a cobertura. A pontuação não considera preço "
                   "justo, seus objetivos, seu perfil de risco nem fatos específicos de cada empresa. "
                   "Cripto não tem balanço, então os critérios de empresa ficam sem dado.")

    with st.expander("📚 O que significa cada critério"):
        for k, rot, melhor, expl in CRITERIOS:
            st.markdown(f"**{rot}** (pontua mais com valor {melhor}): {expl}")
    st.caption("Conteúdo educacional. Não é recomendação de investimento.")


# ------------------------------------------------------------------- aprender
GLOSSARIO = {
    "Ação": "Parte do capital de uma empresa. Quem tem ação é sócio e participa dos lucros e dos riscos do negócio.",
    "Alavancagem": "Operar com valor maior que o capital próprio. Amplifica ganhos, mas também perdas.",
    "Altcoin": "Qualquer criptomoeda que não seja o Bitcoin.",
    "ATR": "Average True Range: medida da amplitude média de variação diária do preço. Muito usada para definir a distância do stop.",
    "Backtest": "Teste de uma regra de investimento em dados históricos. Mostra o passado, não garante o futuro.",
    "Beta": "Sensibilidade de um ativo ao mercado. Beta acima de 1 tende a oscilar mais que o índice de referência.",
    "Blockchain": "Registro digital compartilhado e difícil de adulterar, no qual se baseiam muitas criptomoedas.",
    "Bollinger (bandas)": "Duas faixas ao redor da média móvel, baseadas no desvio padrão. Mostram quanto o preço está esticado em relação à média.",
    "CDI": "Taxa média dos empréstimos entre bancos. Acompanha de perto a Selic e é referência para a renda fixa.",
    "Candle": "Representação gráfica de um período, com preço de abertura, máxima, mínima e fechamento.",
    "Day trade": "Compra e venda do mesmo ativo no mesmo dia.",
    "Dividendo": "Parte do lucro que a empresa distribui aos acionistas.",
    "Dividend Yield (DY)": "Dividendos pagos em 12 meses divididos pelo preço atual do ativo.",
    "Diversificação": "Distribuir o dinheiro entre ativos diferentes para que um problema isolado pese menos no conjunto.",
    "Drawdown": "Queda do ponto mais alto até o ponto mais baixo seguinte, em um período.",
    "EBITDA": "Resultado operacional antes de juros, impostos, depreciação e amortização. Aproxima a geração de caixa da operação.",
    "ETF": "Fundo de índice negociado em bolsa, que busca replicar o desempenho de um índice.",
    "EV/EBITDA": "Valor da empresa (valor de mercado mais dívida líquida) dividido pelo EBITDA.",
    "FII": "Fundo de investimento imobiliário, negociado em bolsa. Costuma distribuir rendimentos periodicamente.",
    "Gap": "Salto de preço entre o fechamento de um pregão e a abertura do seguinte.",
    "Inflação": "Aumento geral dos preços, que reduz o poder de compra do dinheiro ao longo do tempo.",
    "IPCA": "Índice oficial de inflação do Brasil, calculado pelo IBGE.",
    "JCP": "Juros sobre capital próprio: forma de distribuir lucro ao acionista, com tratamento tributário diferente do dividendo. Confira as regras vigentes.",
    "Liquidez": "Facilidade de comprar ou vender um ativo rapidamente sem mexer muito no preço.",
    "Long": "Posição comprada: ganha quando o preço sobe.",
    "MACD": "Indicador que compara duas médias móveis exponenciais para medir a força e a direção do movimento.",
    "Marcação a mercado": "Atualização diária do preço de um título pelo valor de mercado. Vender antes do vencimento pode render menos ou mais que o esperado.",
    "Média móvel": "Média dos preços de fechamento dos últimos N períodos, atualizada a cada novo período.",
    "P/L": "Preço dividido pelo lucro por ação. Indica quantos anos de lucro atual o preço representa.",
    "P/VP": "Preço dividido pelo valor patrimonial por ação.",
    "Payout": "Fração do lucro que a empresa distribui como dividendos.",
    "Pullback": "Recuo do preço dentro de uma tendência maior.",
    "Renda fixa": "Investimentos cuja regra de remuneração é definida na contratação, como títulos públicos e CDBs. Há risco de crédito, de mercado e de inflação.",
    "Renda variável": "Investimentos sem remuneração definida de antemão, como ações, FIIs e criptomoedas.",
    "Rompimento": "Quando o preço ultrapassa um nível de suporte ou resistência relevante.",
    "ROE": "Retorno sobre o patrimônio líquido: lucro dividido pelo capital dos acionistas.",
    "RSI": "Índice de força relativa: oscilador de 0 a 100 que mede a velocidade das altas e das baixas recentes.",
    "Selic": "Taxa básica de juros da economia brasileira, definida pelo Copom do Banco Central.",
    "Short": "Posição vendida: ganha quando o preço cai. O prejuízo potencial é, em teoria, ilimitado.",
    "Small caps": "Empresas de menor valor de mercado, em geral com mais volatilidade e menos liquidez.",
    "Spread": "Diferença entre o melhor preço de compra e o melhor preço de venda de um ativo.",
    "Stablecoin": "Criptomoeda que busca manter valor estável, por exemplo atrelada ao dólar. Tem riscos de lastro e do emissor.",
    "Stop": "Nível de preço definido de antemão para encerrar uma operação e limitar a perda.",
    "Suporte e resistência": "Regiões de preço onde, no passado, a queda ou a alta costumou perder força.",
    "Tesouro Direto": "Programa do Tesouro Nacional que vende títulos públicos federais a pessoas físicas.",
    "Valor de mercado": "Preço da ação multiplicado pelo número total de ações da empresa.",
    "Volatilidade": "Intensidade com que o preço varia. Mais volatilidade significa mais incerteza, para cima e para baixo.",
    "Volume": "Quantidade (ou valor financeiro) negociada em um período.",
}

TRILHAS = [
    {"id": "base", "titulo": "Primeiros passos no mercado", "nivel": "Iniciante", "aulas": [
        {"id": "base1", "titulo": "Os principais tipos de investimento",
         "texto": "Na **renda fixa**, você empresta dinheiro e a regra de remuneração é definida antes. Na **renda variável**, "
                  "o retorno não é conhecido de antemão: **ações** são partes de empresas, **FIIs** são fundos que investem em "
                  "imóveis ou papéis ligados a eles, e **ETFs** são fundos que replicam um índice e são negociados em bolsa. "
                  "**Criptomoedas** são ativos digitais com volatilidade muito alta e regulação ainda em evolução. "
                  "Cada categoria tem riscos próprios.",
         "ferramenta": "Aba Análise → Sobre o ativo"},
        {"id": "base2", "titulo": "Risco, retorno e prazo",
         "texto": "Em geral, a possibilidade de retornos maiores vem junto com a possibilidade de perdas maiores. Retorno passado "
                  "não garante retorno futuro. Três perguntas ajudam a entender um investimento: quanto o preço pode oscilar "
                  "(volatilidade), quão rápido dá para vender (liquidez) e por quanto tempo o dinheiro pode ficar investido "
                  "(prazo). Objetivos e prazos pessoais mudam o que faz sentido para cada pessoa.",
         "ferramenta": "Aba Análise → Gráficos e Tendência e médias"},
        {"id": "base3", "titulo": "Diversificação e concentração",
         "texto": "Concentrar o dinheiro em poucos ativos faz com que um problema isolado pese muito no resultado. "
                  "Diversificar é distribuir entre ativos diferentes (setores, tipos de investimento, países) para reduzir "
                  "esse efeito. Ela não elimina o risco: em crises, muitos ativos caem juntos. Olhar o peso de cada ativo na "
                  "carteira é o primeiro passo.",
         "ferramenta": "Aba Carteira → Distribuição"},
        {"id": "base4", "titulo": "Custos, inflação e impostos",
         "texto": "Corretagem, taxas de custódia, taxas de administração de fundos e impostos reduzem o retorno líquido. "
                  "A inflação reduz o poder de compra: o que importa é o ganho acima dela (juro real). Compare sempre o retorno "
                  "com uma referência, como o CDI ou a inflação. As regras de tributação mudam com o tempo, então confira as "
                  "vigentes antes de decidir.",
         "ferramenta": "Aba Carteira → comparação com Ibovespa e CDI"},
    ], "quiz": [
        {"p": "O que significa dizer que o retorno de um investimento de renda variável não é garantido?",
         "o": ["O valor futuro não é conhecido de antemão e pode ser menor que o investido",
               "Ele sempre rende menos que a renda fixa", "A corretora define o resultado no fim do mês"],
         "c": 0, "e": "Na renda variável não há remuneração definida na contratação."},
        {"p": "O que a diversificação faz?",
         "o": ["Elimina o risco de perda", "Reduz o peso de um problema isolado no conjunto da carteira",
               "Garante retorno maior"],
         "c": 1, "e": "Ela reduz o efeito de eventos isolados, mas não elimina o risco, principalmente em crises."},
        {"p": "Por que comparar o retorno com o CDI ou com a inflação?",
         "o": ["Porque eles definem o preço das ações", "Para saber se o ganho superou uma referência e o poder de compra",
               "Porque são obrigatórios na declaração do imposto"],
         "c": 1, "e": "Sem referência, não dá para saber se o retorno compensou o risco e a inflação."},
    ]},
    {"id": "fund", "titulo": "Analisando uma ação: fundamentos", "nivel": "Intermediário", "aulas": [
        {"id": "fund1", "titulo": "Lendo uma empresa: receita, lucro e dívida",
         "texto": "A **receita** é o que a empresa vende. Descontados custos e despesas, sobra o **lucro**. O **balanço** mostra o "
                  "que a empresa tem (ativos), o que deve (passivos) e o que sobra para os acionistas (**patrimônio líquido**). "
                  "Observe a evolução ao longo de vários períodos, não só o último número. Lucros únicos (venda de um ativo, "
                  "por exemplo) podem distorcer o resultado de um ano.",
         "ferramenta": "Aba Análise → Fundamentos"},
        {"id": "fund2", "titulo": "Múltiplos: P/L, P/VP e EV/EBITDA",
         "texto": "Múltiplos relacionam o preço a algum resultado da empresa. O **P/L** compara preço e lucro, o **P/VP** compara "
                  "preço e patrimônio, e o **EV/EBITDA** compara o valor da empresa, incluindo dívida, com a geração operacional. "
                  "Valores baixos não significam, sozinhos, que algo é barato: setores diferentes têm padrões diferentes, "
                  "empresas cíclicas distorcem o P/L, e bancos exigem outros indicadores. Compare dentro do mesmo setor.",
         "ferramenta": "Aba Comparador"},
        {"id": "fund3", "titulo": "Rentabilidade e endividamento: ROE, margem e dívida",
         "texto": "O **ROE** mede quanto lucro a empresa gera sobre o capital dos acionistas. A **margem líquida** mostra quanto "
                  "sobra de lucro a cada R$ 100 de receita. A **dívida sobre patrimônio** indica o grau de endividamento. "
                  "Um ROE alto obtido com muita dívida carrega mais risco. Vale olhar esses números em conjunto e ao longo do tempo.",
         "ferramenta": "Aba Análise → Fundamentos"},
        {"id": "fund4", "titulo": "Dividendos: yield e payout",
         "texto": "O **dividend yield** relaciona os dividendos dos últimos 12 meses ao preço. O **payout** mostra que parte do "
                  "lucro foi distribuída. Um yield muito alto pode refletir queda forte do preço ou um pagamento que a empresa "
                  "não consegue sustentar. Dividendos não são garantidos e podem ser reduzidos ou suspensos.",
         "ferramenta": "Aba Análise → Fundamentos (histórico de dividendos)"},
    ], "quiz": [
        {"p": "Por que comparar o P/L de empresas de setores muito diferentes pode enganar?",
         "o": ["Porque cada setor tem padrões próprios de lucro, investimento e dívida",
               "Porque o P/L só existe para bancos", "Porque o P/L não muda entre setores"],
         "c": 0, "e": "Os múltiplos fazem mais sentido quando comparados dentro do mesmo setor."},
        {"p": "O que um ROE alto, obtido com muita dívida, indica?",
         "o": ["Risco nenhum, porque o lucro é alto", "Que o retorno veio com mais risco financeiro",
               "Que a empresa não paga dividendos"],
         "c": 1, "e": "A dívida pode ampliar o retorno sobre o patrimônio, mas também aumenta o risco."},
        {"p": "Um dividend yield muito alto pode sinalizar:",
         "o": ["Sempre uma oportunidade", "Possível queda forte do preço ou dividendo difícil de sustentar",
               "Que a empresa não tem lucro nenhum"],
         "c": 1, "e": "O yield sobe quando o preço cai, e o pagamento pode não se repetir."},
    ]},
    {"id": "tec", "titulo": "Análise técnica básica", "nivel": "Intermediário", "aulas": [
        {"id": "tec1", "titulo": "O que a análise técnica é, e o que não é",
         "texto": "A análise técnica estuda o comportamento do preço e do volume no passado para descrever o presente. Ela não "
                  "prevê o futuro e envolve interpretação, o que abre espaço para vieses, como ver apenas o que confirma uma "
                  "opinião prévia. Muitos investidores a usam como contexto, junto com outras análises. Sinais isolados "
                  "costumam falhar.",
         "ferramenta": "Aba Análise → Leitura técnica"},
        {"id": "tec2", "titulo": "Tendência e médias móveis",
         "texto": "A **média móvel** suaviza o preço. Preço acima de uma média longa, como a de 200 dias, costuma ser lido como "
                  "tendência de alta, e abaixo como tendência de baixa. Médias atrasam em relação ao preço, porque são calculadas "
                  "com dados passados, e geram muitos sinais falsos quando o mercado anda de lado.",
         "ferramenta": "Aba Análise → Tendência e médias"},
        {"id": "tec3", "titulo": "RSI, MACD e Bollinger",
         "texto": "O **RSI** mede a força das altas e baixas recentes; leituras acima de 70 ou abaixo de 30 são tratadas como "
                  "extremos, mas em tendências fortes o RSI pode permanecer extremo por muito tempo. O **MACD** compara médias "
                  "exponenciais para medir impulso. As **bandas de Bollinger** mostram quão esticado o preço está em relação à "
                  "média. São descrições do passado recente, não previsões.",
         "ferramenta": "Aba Análise → Gráficos"},
        {"id": "tec4", "titulo": "Suporte, resistência e volume",
         "texto": "**Suporte** e **resistência** são regiões onde o preço já reagiu antes. **Volume** alto em um movimento indica "
                  "mais participação do mercado. Esses níveis não são barreiras garantidas: são referências que podem ser "
                  "rompidas. Por isso, muitos investidores definem de antemão onde a leitura deixa de valer.",
         "ferramenta": "Aba Análise → Leitura técnica (níveis de referência)"},
    ], "quiz": [
        {"p": "A análise técnica serve para:",
         "o": ["Descrever o comportamento do preço, sem prever o futuro", "Garantir a direção do preço",
               "Substituir toda a análise de empresas"],
         "c": 0, "e": "Ela descreve o que o preço fez; qualquer uso para o futuro é interpretação."},
        {"p": "Por que as médias móveis dão muitos sinais falsos quando o mercado anda de lado?",
         "o": ["Porque são calculadas com dados passados e o preço cruza a média várias vezes",
               "Porque só funcionam para cripto", "Porque o volume desaparece"],
         "c": 0, "e": "Sem tendência, o preço cruza a média repetidamente, e cada cruzamento vira um sinal."},
        {"p": "Um RSI acima de 70 significa necessariamente que o preço vai cair?",
         "o": ["Sim, sempre", "Não: em tendências fortes ele pode ficar alto por muito tempo", "Sim, em até 3 dias"],
         "c": 1, "e": "O RSI mede força recente; não determina o que vem depois."},
    ]},
    {"id": "cripto", "titulo": "Criptomoedas: fundamentos e riscos", "nivel": "Iniciante", "aulas": [
        {"id": "cr1", "titulo": "O que são e como funcionam",
         "texto": "Criptomoedas são ativos digitais registrados em uma **blockchain**, um livro-razão compartilhado e difícil de "
                  "adulterar. O **Bitcoin** foi a primeira. Qualquer outra moeda é chamada de **altcoin**. As **stablecoins** "
                  "buscam manter valor estável, por exemplo atrelado ao dólar, e têm riscos ligados ao lastro e ao emissor.",
         "ferramenta": "Aba Mercado cripto"},
        {"id": "cr2", "titulo": "Volatilidade e riscos",
         "texto": "Criptomoedas costumam oscilar muito mais que ações. Há riscos de custódia (perder o acesso às chaves), de "
                  "golpes, de falhas de plataformas, de liquidez em moedas pequenas e de mudanças regulatórias. O mercado "
                  "negocia 24 horas por dia, todos os dias. Entender o quanto uma queda grande pesaria no seu patrimônio "
                  "faz parte da análise.",
         "ferramenta": "Aba Comparador → critérios de volatilidade e maior queda"},
        {"id": "cr3", "titulo": "Indicadores do mercado cripto",
         "texto": "O **valor de mercado** (preço vezes oferta circulante) compara o tamanho das moedas. A **dominância do "
                  "Bitcoin** mostra que fatia do valor total ele representa. O **volume** indica o quanto foi negociado. "
                  "O índice de **medo e ganância** resume o sentimento do mercado: descreve o humor atual, não prevê preços.",
         "ferramenta": "Aba Mercado cripto"},
    ], "quiz": [
        {"p": "Qual destes é um risco específico de quem guarda criptomoedas por conta própria?",
         "o": ["Perder o acesso às chaves de acesso", "Receber dividendos demais", "O mercado fechar às 17h"],
         "c": 0, "e": "Quem perde as chaves pode perder o acesso aos ativos, sem como recuperar."},
        {"p": "O índice de medo e ganância:",
         "o": ["Prevê o preço do próximo mês", "Resume o sentimento atual do mercado", "Garante quando comprar ou vender"],
         "c": 1, "e": "É um indicador de sentimento, não de previsão."},
        {"p": "O que é uma stablecoin?",
         "o": ["Uma criptomoeda que busca manter valor estável, sem garantia de que consiga",
               "Uma ação de empresa de tecnologia", "Um título público"],
         "c": 0, "e": "Ela busca estabilidade, mas depende do lastro e do emissor."},
    ]},
    {
