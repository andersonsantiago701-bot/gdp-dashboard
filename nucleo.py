"""
Santinvest — NÚCLEO: dados, cálculos, salvamento, fontes (Yahoo, CVM, CoinGecko) e conteúdo.
Aqui não há telas. As telas ficam em streamlit_app.py, que importa tudo isto.
Dados do Yahoo Finance via yfinance (sem chave de API). Ferramenta de apoio à pesquisa.
"""

from concurrent.futures import ThreadPoolExecutor
import datetime as dt
from email.utils import parsedate_to_datetime
import html
import io
import json
import re
from urllib.parse import quote_plus
import uuid
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf


def _ler_modo_pessoal():
    """MODO_PESSOAL = True: usa a linguagem avaliativa (veredito). Em False, linguagem neutra
    (para abrir a outras pessoas). Defina em Settings > Secrets do Streamlit: MODO_PESSOAL = false"""
    try:
        valor = st.secrets["MODO_PESSOAL"]
    except Exception:
        return True
    return str(valor).strip().lower() not in ("0", "false", "nao", "não", "no", "off")


MODO_PESSOAL = _ler_modo_pessoal()


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
    altas = sum(1 for _, p, _ in sinais if p > 0)
    baixas = sum(1 for _, p, _ in sinais if p < 0)
    neutros = len(sinais) - altas - baixas

    janela = d.tail(60)
    return {
        "sinais": sinais,
        "total": total,
        "maximo": maximo,
        "veredito": veredito,
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
        veredito = leitura = "Sem dados suficientes"
    else:
        veredito = ("Fundamentos sólidos" if nota >= 0.7
                    else "Fundamentos mistos" if nota >= 0.4 else "Fundamentos fracos")
        leitura = (f"{veredito} · {aprovados} de {len(avaliados)} critérios atendidos" if MODO_PESSOAL
                   else f"{aprovados} de {len(avaliados)} critérios de referência atendidos")
    return {
        "criterios": criterios,
        "aprovados": aprovados,
        "total": len(avaliados),
        "veredito": veredito,
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
        "momento": (lambda m: m["veredito"] if MODO_PESSOAL else m["curto"])(avaliar_momento(d)),
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


@st.cache_data(ttl=300, show_spinner=False)
def variacoes_eua(tickers):
    return _buscar_variacoes(tickers, "", "US$")


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


def filtrar_glossario(busca):
    busca = (busca or "").strip().lower()
    itens = sorted(GLOSSARIO.items(), key=lambda kv: kv[0].lower())
    if not busca:
        return itens
    return [(t, d) for t, d in itens if busca in t.lower() or busca in d.lower()]


# ----------------------------------------------------- CVM (dados oficiais)
# Demonstrações padronizadas (DFP) e cadastro (FCA) do Portal Dados Abertos da CVM.
# Licença dos dados: ODbL (cite a fonte). Valores do último exercício anual, consolidados.
CVM_DOC = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC"
CVM_CONTAS = {"3.01", "3.11", "2.03", "1.01", "1.01.01", "1.01.02", "2.01", "2.01.04", "2.02.01"}
CVM_ESCALA = {"MIL": 1e3, "UNIDADE": 1.0, "MILHAO": 1e6, "MILHÃO": 1e6}


def _digitos(s):
    return re.sub(r"\D", "", str(s or ""))


def _csv_do_zip(conteudo, trecho_nome):
    """Lê um CSV (latin-1, separado por ';') de dentro de um ZIP em memória."""
    with zipfile.ZipFile(io.BytesIO(conteudo)) as z:
        nome = next((n for n in z.namelist() if trecho_nome.lower() in n.lower()), None)
        if nome is None:
            return None
        with z.open(nome) as f:
            df = pd.read_csv(f, sep=";", encoding="latin-1", dtype=str)
    df.columns = [c.strip().upper() for c in df.columns]
    return df


@st.cache_data(ttl=7 * 86400, show_spinner=False)
def cvm_mapa_tickers(ano):
    """código de negociação (ex.: PETR4) -> CNPJ, a partir do Formulário Cadastral (FCA)."""
    try:
        r = requests.get(f"{CVM_DOC}/FCA/DADOS/fca_cia_aberta_{ano}.zip", headers=UA, timeout=120)
        r.raise_for_status()
        df = _csv_do_zip(r.content, "valor_mobiliario")
    except Exception:
        return None
    if df is None:
        return None
    c_cnpj = next((c for c in df.columns if "CNPJ" in c), None)
    c_cod = next((c for c in df.columns if "CODIGO" in c and "NEGOCIA" in c), None)
    if not c_cnpj or not c_cod:
        return None
    mapa = {}
    for cnpj, cods in zip(df[c_cnpj], df[c_cod]):
        for cod in re.split(r"[;,/\s]+", str(cods or "").upper()):
            if re.fullmatch(r"[A-Z]{4}\d{1,2}", cod):
                mapa[cod] = _digitos(cnpj)
    return mapa


@st.cache_data(ttl=7 * 86400, show_spinner=False)
def cvm_contas(ano):
    """Contas consolidadas (DRE, ativo e passivo) do DFP do ano, só as que usamos."""
    try:
        r = requests.get(f"{CVM_DOC}/DFP/DADOS/dfp_cia_aberta_{ano}.zip", headers=UA, timeout=180)
        r.raise_for_status()
    except Exception:
        return None
    partes = []
    for tag in ("_DRE_con_", "_BPA_con_", "_BPP_con_"):
        try:
            df = _csv_do_zip(r.content, tag)
        except Exception:
            df = None
        if df is None:
            continue
        faltam = {"CNPJ_CIA", "DENOM_CIA", "DT_REFER", "ORDEM_EXERC", "CD_CONTA", "DS_CONTA", "VL_CONTA"} - set(df.columns)
        if faltam:
            continue
        df = df[df["CD_CONTA"].isin(CVM_CONTAS)].copy()
        escala = df["ESCALA_MOEDA"].str.upper().map(CVM_ESCALA).fillna(1.0) if "ESCALA_MOEDA" in df else 1.0
        df["VALOR"] = pd.to_numeric(df["VL_CONTA"], errors="coerce") * escala
        df["CNPJ"] = df["CNPJ_CIA"].map(_digitos)
        df["ORDEM"] = np.where(df["ORDEM_EXERC"].str.upper().str.startswith("PEN"), "PEN", "ULT")
        df["VERSAO"] = pd.to_numeric(df["VERSAO"], errors="coerce").fillna(0) if "VERSAO" in df else 0
        partes.append(df[["CNPJ", "DENOM_CIA", "DT_REFER", "VERSAO", "ORDEM", "CD_CONTA", "DS_CONTA", "VALOR"]])
    return pd.concat(partes, ignore_index=True) if partes else None


def calcular_fundamentos_cvm(df):
    """df: contas de UMA empresa (saída de cvm_contas). Usa o último exercício anual disponível."""
    ult = df[df["ORDEM"] == "ULT"]
    if ult.empty:
        return {"erro": "Sem demonstrações do último exercício."}
    ref = ult["DT_REFER"].max()
    doc = df[df["DT_REFER"] == ref]
    doc = doc[doc["VERSAO"] == doc["VERSAO"].max()]  # versão mais recente (reapresentações)
    cur, prev = doc[doc["ORDEM"] == "ULT"], doc[doc["ORDEM"] == "PEN"]

    def v(sub, codigo, ds=None):
        x = sub[sub["CD_CONTA"] == codigo]
        if ds:
            x = x[x["DS_CONTA"].str.contains(ds, case=False, na=False, regex=True)]
        x = x["VALOR"].dropna()
        return float(x.iloc[0]) if not x.empty else None

    def soma(*valores):
        validos = [x for x in valores if x is not None]
        return sum(validos) if validos else None

    receita, receita_ant = v(cur, "3.01", "receita"), v(prev, "3.01", "receita")
    lucro = v(cur, "3.11", "lucro|preju")
    pl = v(cur, "2.03", "patrim")
    divida = soma(v(cur, "2.01.04"), v(cur, "2.02.01"))
    caixa = soma(v(cur, "1.01.01"), v(cur, "1.01.02"))
    ac, pc = v(cur, "1.01"), v(cur, "2.01")
    dl = divida - caixa if divida is not None and caixa is not None else None

    def razao(a, b, mult=1.0):
        return a / b * mult if a is not None and b not in (None, 0) else None

    return {
        "nome": str(doc["DENOM_CIA"].iloc[0]) if not doc.empty else "", "ref": ref,
        "exercicio": str(ref)[:4], "receita": receita, "lucro": lucro, "pl": pl,
        "divida_liquida": dl,
        "margem": razao(lucro, receita, 100), "roe": razao(lucro, pl, 100),
        "cresc": (receita / receita_ant - 1) * 100 if receita is not None and receita_ant else None,
        "div_liq_pl": razao(dl, pl, 100), "liq_corrente": razao(ac, pc)}


def cvm_fundamentos(ticker):
    """Fundamentos do último exercício anual (DFP) a partir da CVM. Devolve {'erro': ...} se falhar."""
    base = ticker.upper().replace(".SA", "")
    hoje = dt.date.today()
    cnpj = None
    for ano in (hoje.year, hoje.year - 1, hoje.year - 2):
        mapa = cvm_mapa_tickers(ano)
        if mapa and base in mapa:
            cnpj = mapa[base]
            break
    if not cnpj:
        return {"erro": "Não achei esse código no cadastro da CVM (FCA). FIIs e ETFs não usam esse formulário."}
    frames = [cvm_contas(a) for a in (hoje.year - 1, hoje.year)]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return {"erro": "Não consegui baixar as demonstrações da CVM agora. Tente de novo mais tarde."}
    df = pd.concat(frames, ignore_index=True)
    df = df[df["CNPJ"] == cnpj]
    if df.empty:
        return {"erro": "A CVM não trouxe demonstrações consolidadas dessa empresa nos arquivos recentes."}
    return calcular_fundamentos_cvm(df)


def comparar_cvm_yahoo(cv, info):
    """Tabela lado a lado: CVM (último exercício anual) x Yahoo (12 meses / último trimestre)."""
    def mi(x):
        return None if x is None else x / 1e6

    def pc(x):
        return None if x is None else x * 100

    linhas = [
        ("Receita líquida (R$ mi)", mi(cv["receita"]), mi(num(info, "totalRevenue")), False),
        ("Lucro líquido (R$ mi)", mi(cv["lucro"]), mi(num(info, "netIncomeToCommon")), False),
        ("Margem líquida (%)", cv["margem"], pc(num(info, "profitMargins")), True),
        ("ROE (%)", cv["roe"], pc(num(info, "returnOnEquity")), True),
        ("Crescimento da receita (%)", cv["cresc"], pc(num(info, "revenueGrowth")), True),
        ("Dívida líquida / patrimônio (%)", cv["div_liq_pl"], None, False),
        ("Liquidez corrente", cv["liq_corrente"], num(info, "currentRatio"), False),
    ]
    return pd.DataFrame([
        {"Indicador": n, f"CVM ({cv['exercicio']})": a, "Yahoo": b,
         "Diferença (p.p.)": (a - b) if pp and a is not None and b is not None else None}
        for n, a, b, pp in linhas])
