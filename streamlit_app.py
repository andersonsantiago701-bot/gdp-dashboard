"""
Santinvest — app Streamlit
Dados: Yahoo Finance via yfinance (sem chave de API).
Ferramenta de apoio à pesquisa. Não é recomendação de compra ou venda.
"""

import datetime as dt
import html
import json
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

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
    if total >= 4:
        veredito = "Favorável"
    elif total >= 1:
        veredito = "Levemente favorável"
    elif total >= -1:
        veredito = "Neutro"
    elif total >= -3:
        veredito = "Cautela"
    else:
        veredito = "Desfavorável"

    janela = d.tail(60)
    return {
        "sinais": sinais,
        "total": total,
        "maximo": maximo,
        "veredito": veredito,
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
    elif nota >= 0.7:
        leitura = "Fundamentos sólidos"
    elif nota >= 0.4:
        leitura = "Fundamentos mistos"
    else:
        leitura = "Fundamentos fracos"
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
    return {"favoritos": [], "alertas": []}


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
        "momento": avaliar_momento(d)["veredito"],
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
def url_google_noticias(consulta):
    return ("https://news.google.com/rss/search?q=" + quote_plus(consulta + " when:2d")
            + "&hl=pt-BR&gl=BR&ceid=BR:pt-419")


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
        itens.append({"ti
