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


# --------------------------------------------------------------------- app
def render_analise(escolha, periodo):
    ticker = escolha["simbolo"]
    with st.spinner(f"Buscando dados de {ticker}..."):
        hist = carregar_historico(ticker)
        info = carregar_info(ticker)
        divs = carregar_dividendos(ticker)

    if hist is None or hist.empty or len(hist) < 30:
        st.error(f"Não encontrei histórico suficiente para {ticker}.")
        return

    cripto = escolha["tipo"] == "CRYPTOCURRENCY" or info.get("quoteType") == "CRYPTOCURRENCY"
    dias_ano = 365 if cripto else 252  # cripto negocia todos os dias

    d = calcular_indicadores(hist)
    u = d.iloc[-1]
    ant = d.iloc[-2]
    preco = float(u["Close"])
    variacao = (preco / float(ant["Close"]) - 1) * 100
    moeda = info.get("currency", "USD" if cripto else "BRL")
    simbolo = {"BRL": "R$", "USD": "US$"}.get(moeda, moeda)

    nome = info.get("longName") or info.get("shortName") or escolha["nome"]
    st.subheader(f"{nome} ({ticker})")
    if info.get("sector"):
        st.caption(f"{info.get('sector')} · {info.get('industry', '')}")

    janela_52 = d.tail(dias_ano)
    max52, min52 = float(janela_52["High"].max()), float(janela_52["Low"].min())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Preço", f"{simbolo} {br(preco)}", f"{br(variacao)}%")
    c2.metric("Máx. 12 meses", f"{simbolo} {br(max52)}",
              f"{br((preco / max52 - 1) * 100, 1)}% do topo", delta_color="off")
    c3.metric("Mín. 12 meses", f"{simbolo} {br(min52)}",
              f"{br((preco / min52 - 1) * 100, 1)}% do fundo", delta_color="off")
    c4.metric("Valor de mercado", grande(num(info, "marketCap")))

    n_dias = {"6 meses": dias_ano // 2, "1 ano": dias_ano,
              "2 anos": dias_ano * 2, "5 anos": len(d)}[periodo]
    visivel = d.tail(n_dias)

    tendencia = analisar_tendencia(d)
    momento = avaliar_momento(d)
    fund = None if cripto else avaliar_fundamentos(
        info, dividend_yield_12m(divs, preco))

    aba_res, aba_graf, aba_fund, aba_tend, aba_mom = st.tabs(
        ["Resumo", "Gráficos", "Fundamentos", "Tendência e médias", "Momento de compra"])

    with aba_res:
        a, b, c = st.columns(3)
        if fund:
            a.metric("Fundamentos", fund["leitura"],
                     f"{fund['aprovados']}/{fund['total']} critérios", delta_color="off")
        else:
            a.metric("Fundamentos", "Não se aplica", "criptomoeda", delta_color="off")
        b.metric("Tendência", tendencia["geral"].replace("Tendência de ", "").replace("Tendência ", ""))
        c.metric("Momento técnico", momento["veredito"],
                 f"{momento['total']:+d} de ±{momento['maximo']}", delta_color="off")
        st.write(
            "O resumo cruza três leituras independentes: **valor** (fundamentos), "
            "**direção** (tendência) e **timing** (sinais técnicos). "
            "Um ativo pode ter bons fundamentos e mau momento técnico, e vice-versa. "
            "Veja o detalhe nas outras abas.")
        if cripto:
            st.warning("Criptomoedas são muito mais voláteis que ações e não têm balanço "
                       "nem lucro. A análise aqui é só técnica (preço e volume).")
        alvo = num(info, "targetMeanPrice")
        if alvo:
            st.info(
                f"Preço-alvo médio dos analistas (Yahoo): {simbolo} {br(alvo)} "
                f"({br((alvo / preco - 1) * 100, 1)}% vs. preço atual). "
                "Projeções de analistas costumam errar e variam bastante.")

    with aba_graf:
        st.plotly_chart(grafico_principal(visivel), use_container_width=True)

    with aba_fund:
        if cripto:
            st.markdown("### Criptomoeda: sem fundamentos tradicionais")
            st.write("Não há P/L, ROE ou dividendos. Alguns dados de mercado:")
            x1, x2, x3 = st.columns(3)
            x1.metric("Valor de mercado", grande(num(info, "marketCap")))
            x2.metric("Oferta circulante", grande(num(info, "circulatingSupply")))
            x3.metric("Volume 24h", grande(num(info, "volume24Hr") or float(u["Volume"])))
        else:
            st.markdown(f"### {fund['leitura']}  ·  {fund['aprovados']} de {fund['total']} critérios atendidos")
            tabela = pd.DataFrame(
                [{"Indicador": n, "Valor": v, "Referência": r,
                  "Resultado": "✅" if ok else ("❌" if ok is False else "➖ sem dado")}
                 for n, v, r, ok in fund["criterios"]])
            st.dataframe(tabela, hide_index=True, use_container_width=True)
            st.caption("Limites genéricos de referência. Bancos, seguradoras, FIIs e empresas de "
                       "crescimento têm padrões próprios; compare sempre com o setor.")
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Lucro por ação (LPA)", br(num(info, "trailingEps")))
            x2.metric("Valor patrimonial/ação", br(num(info, "bookValue")))
            x3.metric("Receita (12m)", grande(num(info, "totalRevenue")))
            x4.metric("Beta", br(num(info, "beta")))
            if divs is not None and not divs.empty:
                st.plotly_chart(grafico_dividendos(divs), use_container_width=True)

    with aba_tend:
        st.markdown(f"### {tendencia['geral']}")
        for prazo, rot, txt in tendencia["itens"]:
            st.write(f"**{prazo}:** {rot}. {txt}")
        st.write(f"**Cruzamento de médias:** {tendencia['cruzamento']}")
        if tendencia["inclinacao"] is not None:
            st.write(f"**Inclinação da média de 50 dias (último mês):** "
                     f"{pct(tendencia['inclinacao'])}")
        linhas = []
        for n in (20, 50, 200):
            m = u[f"SMA{n}"]
            linhas.append({
                "Média": f"Simples {n} dias",
                "Valor": br(m) if pd.notna(m) else "—",
                "Preço vs. média": pct((preco / m - 1) * 100) if pd.notna(m) else "—"})
        m_ano = float(janela_52["Close"].mean())
        linhas.append({"Média": "Último ano", "Valor": br(m_ano),
                       "Preço vs. média": pct((preco / m_ano - 1) * 100)})
        st.dataframe(pd.DataFrame(linhas), hide_index=True, use_container_width=True)
        st.markdown("#### Volume de negociação")
        v1, v2, v3 = st.columns(3)
        v1.metric("Último dia", grande(float(u["Volume"])))
        v2.metric("Média 20 dias", grande(float(u["VOL_MM20"])))
        v3.metric("Média 1 ano", grande(float(janela_52["Volume"].mean())))

    with aba_mom:
        st.markdown(f"### Momento técnico: {momento['veredito']} "
                    f"({momento['total']:+d} de ±{momento['maximo']})")
        st.dataframe(
            pd.DataFrame([{"Sinal": n, "Pontos": f"{p:+d}", "Leitura": t}
                          for n, p, t in momento["sinais"]]),
            hide_index=True, use_container_width=True)
        st.markdown("#### Níveis para observar")
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Suporte (60 dias)", f"{simbolo} {br(momento['suporte'])}")
        n2.metric("Resistência (60 dias)", f"{simbolo} {br(momento['resistencia'])}")
        n3.metric("Banda inferior Bollinger", f"{simbolo} {br(momento['bb_inf'])}")
        n4.metric("Média 50 dias", f"{simbolo} {br(momento['sma50'])}")
        st.write(
            "Regiões próximas ao suporte, à banda inferior ou à média de 50 dias costumam ser "
            "pontos de observação para entradas, ao passo que rompimentos abaixo do suporte "
            "indicam que a leitura pode estar errada.")
        st.warning(
            "O placar mede apenas o momento técnico, com regras simples e conhecidas. "
            "Ele não prevê preços e ignora notícias, resultados e o seu perfil de risco. "
            "Não é recomendação de investimento.")


def render_favoritos(dados):
    favs = dados["favoritos"]
    if not favs:
        st.info("Você ainda não tem favoritos. Escolha um ativo e toque em "
                "☆ Favoritar na barra lateral.")
        return
    linhas = []
    for f in favs:
        r = resumo_ativo(f["simbolo"])
        m = moeda_de(f["simbolo"])
        linhas.append({
            "Ativo": f["nome"], "Código": f["simbolo"],
            "Preço": f"{m} {br(r['preco'])}" if r else "—",
            "Dia": pct(r["dia"]) if r else "—",
            "RSI": br(r["rsi"], 0) if r and r["rsi"] is not None else "—",
            "Tendência": r["tendencia"] if r else "—",
            "Momento técnico": r["momento"] if r else "—"})
    st.dataframe(pd.DataFrame(linhas), hide_index=True, use_container_width=True)
    st.caption("Para abrir a análise completa, escolha 'Favoritos' na barra lateral.")


def render_alertas(dados, escolha, persistente):
    st.markdown("### Criar alerta")
    if not persistente:
        st.warning("O salvamento ainda não está configurado: seus alertas somem ao recarregar "
                   "a página e nenhuma notificação é enviada. Veja o LEIA-ME-ALERTAS.md.")
    else:
        st.caption("As notificações no celular são enviadas pelo verificador automático "
                   "(passo 5 do LEIA-ME-ALERTAS.md), a cada ~15 minutos.")

    if escolha is None:
        st.info("Escolha um ativo na barra lateral para criar um alerta para ele.")
    else:
        r = resumo_ativo(escolha["simbolo"])
        m = moeda_de(escolha["simbolo"])
        if r is None:
            st.info("Não consegui o preço atual deste ativo agora.")
        else:
            st.write(f"**{escolha['nome']}** ({escolha['simbolo']}) · preço atual "
                     f"{m} {br(r['preco'])}"
                     + (f" · RSI {br(r['rsi'], 0)}" if r["rsi"] is not None else ""))
            tipo = st.selectbox("Avisar quando", list(TIPOS_ALERTA),
                                format_func=lambda t: TIPOS_ALERTA[t],
                                key="tipo_alerta")
            if tipo == "rsi_abaixo":
                valor = st.number_input("RSI igual ou abaixo de", min_value=1.0,
                                        max_value=99.0, value=30.0, step=1.0, key="v_rsi")
            else:
                padrao = round(r["preco"] * (0.95 if tipo == "preco_abaixo" else 1.05), 2)
                valor = st.number_input(f"Preço ({m})", min_value=0.0, value=padrao,
                                        step=0.01, format="%.2f", key=f"v_{tipo}_{escolha['simbolo']}")
            st.button("Criar alerta", on_click=criar_alerta, args=(escolha, tipo, valor),
                      type="primary")
            if tipo == "rsi_abaixo":
                st.caption("RSI abaixo de 30 costuma indicar queda exagerada (sobrevendido). "
                           "É um sinal de observação, não garantia de reversão.")

    st.markdown("### Meus alertas")
    if not dados["alertas"]:
        st.caption("Nenhum alerta criado.")
        return
    for a in dados["alertas"]:
        m = moeda_de(a["simbolo"])
        alvo = br(a["valor"], 0) if a["tipo"] == "rsi_abaixo" else f"{m} {br(a['valor'])}"
        if a.get("disparado_em"):
            estado = "🔔 disparado em " + a["disparado_em"][:16].replace("T", " ") + " UTC"
        else:
            estado = "⏳ ativo"
        col1, col2, col3 = st.columns([6, 2, 2])
        col1.write(f"**{a['nome']}** ({a['simbolo']}): {TIPOS_ALERTA[a['tipo']]} {alvo} · {estado}")
        if a.get("disparado_em"):
            col2.button("Reativar", key=f"re_{a['id']}", on_click=reativar_alerta, args=(a["id"],))
        col3.button("Excluir", key=f"ex_{a['id']}", on_click=remover_alerta, args=(a["id"],))


def render_noticias(dados, escolha):
    topo1, topo2 = st.columns([3, 1])
    topo1.caption("Manchetes de InfoMoney e Google Notícias. Atualiza sozinho a cada 5 minutos.")
    if topo2.button("🔄 Atualizar"):
        ler_noticias.clear()
    opcoes = ["Mercado em geral"]
    if escolha:
        opcoes.append("Ativo selecionado")
    if dados["favoritos"]:
        opcoes.append("Meus favoritos")
    foco = st.radio("Mostrar", opcoes, horizontal=True)

    if foco == "Mercado em geral":
        listas = [ler_noticias("https://www.infomoney.com.br/feed/", "InfoMoney"),
                  ler_noticias(url_google_noticias("mercado financeiro bolsa B3 dólar"),
                               "Google Notícias")]
        mostrar_noticias(juntar_noticias(listas, 25))
    elif foco == "Ativo selecionado":
        st.write(f"Notícias sobre **{escolha['nome']}**")
        lista = ler_noticias(url_google_noticias(consulta_do_ativo(escolha)), "Google Notícias")
        mostrar_noticias(juntar_noticias([lista], 15))
    else:
        for f in dados["favoritos"][:6]:
            st.markdown(f"#### {f['nome']}")
            lista = ler_noticias(url_google_noticias(consulta_do_ativo(f)), "Google Notícias")
            mostrar_noticias(juntar_noticias([lista], 3))
        if len(dados["favoritos"]) > 6:
            st.caption("Mostrando os 6 primeiros favoritos.")


def main():
    dados, persistente = ler_dados()
    barra_cotacoes(dados)
    st.title("📈 Santinvest")
    st.caption("Ações, FIIs, ETFs e criptomoedas. Dados do Yahoo Finance. "
               "Ferramenta de apoio à pesquisa, não é recomendação de compra ou venda.")

    erro = st.session_state.pop("erro_salvar", None)
    if erro:
        st.error(erro)

    escolha = None
    with st.sidebar:
        st.header("Ativo")
        modo = "Buscar"
        if dados["favoritos"]:
            modo = st.radio("Escolher por", ["Buscar", "Favoritos"], horizontal=True)
        if modo == "Buscar":
            texto = st.text_input("Nome ou código", value="Petrobras",
                                  help="Ex.: Petrobras, Vale, Apple, HGLG11, Bitcoin, Ethereum")
            resultados = buscar_ativos(texto)
            if resultados:
                i = st.selectbox("Escolha o ativo", range(len(resultados)),
                                 format_func=lambda k: resultados[k]["rotulo"])
                escolha = resultados[i]
            else:
                st.error("Não encontrei esse ativo. Tente outro nome ou o código "
                         "(ex.: PETR4, BTC-USD).")
        else:
            favs = dados["favoritos"]
            i = st.selectbox("Seus favoritos", range(len(favs)),
                             format_func=lambda k: f"{favs[k]['nome']} ({favs[k]['simbolo']})")
            escolha = favs[i]
        periodo = st.selectbox("Período dos gráficos", ["6 meses", "1 ano", "2 anos", "5 anos"], index=1)
        if escolha:
            ja = any(f["simbolo"] == escolha["simbolo"] for f in dados["favoritos"])
            st.button("★ Remover dos favoritos" if ja else "☆ Favoritar",
                      on_click=alternar_favorito, args=(escolha,), use_container_width=True)
        st.caption("💾 Favoritos e alertas salvos." if persistente else
                   "⚠️ Salvamento não configurado: favoritos e alertas somem ao recarregar.")

    for a, r in alertas_atingidos(dados):
        m = moeda_de(a["simbolo"])
        st.warning(f"🔔 Alerta atingido: {a['nome']} ({a['simbolo']}) está em "
                   f"{m} {br(r['preco'])} · {TIPOS_ALERTA[a['tipo']]} "
                   + (br(a['valor'], 0) if a['tipo'] == 'rsi_abaixo' else f"{m} {br(a['valor'])}"))

    t_an, t_fav, t_al, t_nt = st.tabs(["Análise", "⭐ Favoritos", "🔔 Alertas", "📰 Notícias"])
    with t_an:
        if escolha:
            render_analise(escolha, periodo)
        else:
            st.info("Digite o nome ou código de um ativo na barra lateral.")
    with t_fav:
        render_favoritos(dados)
    with t_al:
        render_alertas(dados, escolha, persistente)
    with t_nt:
        render_noticias(dados, escolha)


if __name__ == "__main__":
    main()

