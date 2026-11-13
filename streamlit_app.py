"""
Santinvest — TELAS (interface). Os dados e os cálculos ficam em nucleo.py.
Ferramenta de apoio à pesquisa. Não é recomendação de compra ou venda.
"""

import datetime as dt
import math
import re
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Santinvest", page_icon="📈", layout="wide")

from nucleo import (
    BT_MAX_BARRAS, BT_SETUPS, CRITERIOS, FEEDS_CRIPTO_EN, MAX_COMPARAR, MODO_PESSOAL,
    PRESETS_CRIT, ROTULO_CRIT, TEMAS_CRIPTO, TIPOS_ALERTA, UNIVERSO_B3, UNIVERSO_EUA,
    alertas_atingidos, alternar_favorito, analisar_tendencia, atr_diario,
    avaliar_fundamentos, avaliar_momento, barra_cotacoes, br, bt_preparar, bt_simular,
    bt_sinais, buscar_ativos, calcular_indicadores, carregar_dividendos,
    carregar_historico, carregar_info, cmp_adicionar_busca, cmp_incluir, cmp_preset,
    cmp_todos_iguais, coingecko_info, comparar_cvm_yahoo, consulta_do_ativo,
    criar_alerta, cripto_global, cripto_mercado, cvm_fundamentos, dividend_yield_12m,
    eh_cripto, estatisticas_diario, filtrar_glossario, fx_usdbrl, grafico_dividendos,
    grafico_principal, grande, juntar_noticias, ler_dados, ler_noticias, limpar_wiki,
    medo_ganancia, metricas_ativo, moeda_de, montar_carteira, montar_diario,
    mostrar_noticias, mutar, nome_curto, num, pct, pontuar, ranking_altas_baixas,
    reativar_alerta, remover_alerta, remover_operacao, remover_posicao, resumo_ativo,
    salvar_operacao, salvar_posicao, separar_wiki, trajetoria, url_google_noticias,
    variacoes_b3, variacoes_eua, wiki_buscar, wiki_texto,
)


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

CONFIG_EUA = {
    "Preço (US$)": st.column_config.NumberColumn(format="%.2f"),
    "Variação (%)": st.column_config.NumberColumn(format="%+.2f%%"),
    "Volume (US$ mi)": st.column_config.NumberColumn(format="%.1f"),
}


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
    {"id": "risco", "titulo": "Gestão de risco e comportamento", "nivel": "Todos os níveis", "aulas": [
        {"id": "rk1", "titulo": "Quanto arriscar em cada operação",
         "texto": "Uma regra comum de gestão de risco é definir, antes de entrar, quanto do capital você aceita perder se o plano "
                  "der errado. A quantidade sai da conta: **valor em risco ÷ distância até o stop**. Exemplo: capital de "
                  "R$ 10.000, risco de 1% (R$ 100) e stop R$ 2 abaixo da entrada resultam em 50 unidades. O percentual de risco "
                  "é uma escolha pessoal.",
         "ferramenta": "Aba Risco → Calculadora de posição"},
        {"id": "rk2", "titulo": "Taxa de acerto, payoff e expectativa",
         "texto": "Acertar pouco pode ser compatível com resultado positivo, e acertar muito pode não ser. Tudo depende do "
                  "**payoff** (ganho médio ÷ perda média). Exemplo: ganhar em 40% das vezes, com ganhos de 2R e perdas de 1R, "
                  "dá uma expectativa de 0,4 × 2 − 0,6 × 1 = +0,2R por operação. Para saber os seus números, é preciso "
                  "registrar tudo, inclusive os prejuízos.",
         "ferramenta": "Aba Risco → Diário de operações e Backtest"},
        {"id": "rk3", "titulo": "Vieses comportamentais",
         "texto": "Decisões financeiras sofrem com vieses: **excesso de confiança**, **aversão à perda** (segurar o que cai para "
                  "não realizar o prejuízo), **efeito manada**, medo de ficar de fora (FOMO) e **viés de confirmação**. "
                  "Práticas que ajudam: plano escrito antes de agir, registro das decisões e revisão periódica.",
         "ferramenta": "Aba Risco → Diário de operações"},
        {"id": "rk4", "titulo": "Golpes e promessas de ganho fácil",
         "texto": "Desconfie de rentabilidade garantida acima do normal, pressa para decidir, pagamento por indicação, "
                  "'robôs' que prometem lucro certo e contas de desconhecidos que 'devolvem' mais dinheiro. Antes de entregar "
                  "dinheiro a uma empresa, consulte se ela é autorizada nos sites da CVM e do Banco Central.",
         "ferramenta": "—"},
    ], "quiz": [
        {"p": "Capital de R$ 20.000, risco de 1% e stop R$ 4 abaixo da entrada. Quantas unidades a conta indica?",
         "o": ["50", "5", "500"], "c": 0, "e": "R$ 200 de risco ÷ R$ 4 por unidade = 50 unidades."},
        {"p": "Ganhar em 40% das operações, com ganhos de 2R e perdas de 1R, resulta em expectativa:",
         "o": ["Negativa", "Positiva, de +0,2R por operação", "Nula"],
         "c": 1, "e": "0,4 × 2 − 0,6 × 1 = +0,2R. Sem custos, claro."},
        {"p": "Qual destes é um sinal de alerta de golpe?",
         "o": ["Divulgação clara dos riscos", "Promessa de rentabilidade garantida acima do normal e pressa para decidir",
               "Empresa autorizada na CVM"],
         "c": 1, "e": "Promessa de ganho garantido e pressão para decidir são sinais clássicos."},
    ]},
]
ID_AULAS = {a["id"] for t in TRILHAS for a in t["aulas"]}


def _progresso(d):
    p = d.setdefault("progresso", {})
    p.setdefault("aulas", [])
    p.setdefault("quiz", {})
    return p


def salvar_aula(aula_id):
    marcado = bool(st.session_state.get(f"aula_{aula_id}"))

    def fn(d):
        aulas = _progresso(d)["aulas"]
        if marcado and aula_id not in aulas:
            aulas.append(aula_id)
        if not marcado and aula_id in aulas:
            aulas.remove(aula_id)
    mutar(fn)


def corrigir_quiz(trilha_id):
    ss = st.session_state
    trilha = next(t for t in TRILHAS if t["id"] == trilha_id)
    certos = [ss.get(f"quiz_{trilha_id}_{i}") == q["o"][q["c"]] for i, q in enumerate(trilha["quiz"])]
    ss[f"quiz_ok_{trilha_id}"] = certos

    def fn(d):
        _progresso(d)["quiz"][trilha_id] = {"acertos": sum(certos), "total": len(certos)}
    mutar(fn)


def render_aprender(dados):
    prog = dados.get("progresso", {})
    feitas = set(prog.get("aulas", [])) & ID_AULAS
    st.markdown("### 📚 Aprender")
    st.caption("Conteúdo educacional sobre como analisar investimentos. Não considera a sua situação "
               "pessoal e não é recomendação. Para orientação personalizada, procure um profissional habilitado.")
    st.progress(len(feitas) / len(ID_AULAS), text=f"{len(feitas)} de {len(ID_AULAS)} aulas concluídas")

    aba_tr, aba_gl = st.tabs(["Trilhas de estudo", "Glossário"])
    with aba_tr:
        for t in TRILHAS:
            ids = [a["id"] for a in t["aulas"]]
            n = len([i for i in ids if i in feitas])
            with st.expander(f"{t['titulo']} · {t['nivel']} · {n}/{len(ids)} aulas"):
                for a in t["aulas"]:
                    st.markdown(f"#### {a['titulo']}")
                    st.markdown(a["texto"].replace("$", "\\$"))
                    if a["ferramenta"] != "—":
                        st.caption(f"🔧 Veja no app: {a['ferramenta']}")
                    st.checkbox("Marcar como concluída", value=a["id"] in feitas, key=f"aula_{a['id']}",
                                on_change=salvar_aula, args=(a["id"],))
                st.markdown("#### ✍️ Quiz da trilha")
                for i, q in enumerate(t["quiz"]):
                    st.radio(f"{i + 1}. {q['p']}", q["o"], index=None, key=f"quiz_{t['id']}_{i}")
                st.button("Corrigir", key=f"corrigir_{t['id']}", on_click=corrigir_quiz, args=(t["id"],))
                certos = st.session_state.get(f"quiz_ok_{t['id']}")
                if certos is not None:
                    for i, (q, ok) in enumerate(zip(t["quiz"], certos)):
                        st.markdown(f"{'✅' if ok else '❌'} **{i + 1}.** {q['e']}")
                    st.caption(f"Resultado: {sum(certos)} de {len(certos)}.")
                elif t["id"] in prog.get("quiz", {}):
                    r = prog["quiz"][t["id"]]
                    st.caption(f"Último resultado salvo: {r['acertos']} de {r['total']}.")
        st.caption("Para ir além: Portal do Investidor (CVM), B3 Educação e o site do Tesouro Direto "
                   "têm material gratuito.")
    with aba_gl:
        busca = st.text_input("Buscar termo", key="gloss_busca")
        itens = filtrar_glossario(busca)
        if not itens:
            st.info("Nenhum termo encontrado.")
        for termo, definicao in itens:
            st.markdown(f"**{termo}**: {definicao}")


# --------------------------------------------------------------------- app
def render_cvm(cv, info):
    if cv.get("erro"):
        st.info(cv["erro"])
        return
    st.markdown(f"**{cv['nome']}** · exercício encerrado em {str(cv['ref'])[:10]} (DFP, consolidado)")
    tabela = comparar_cvm_yahoo(cv, info)
    col_cvm = f"CVM ({cv['exercicio']})"
    st.dataframe(tabela, hide_index=True, use_container_width=True, column_config={
        col_cvm: st.column_config.NumberColumn(format="%.2f"),
        "Yahoo": st.column_config.NumberColumn(format="%.2f"),
        "Diferença (p.p.)": st.column_config.NumberColumn(format="%+.1f")})
    st.caption("Fonte: CVM, Portal Dados Abertos (licença ODbL). A CVM mostra o último exercício anual; "
               "o Yahoo usa os últimos 12 meses ou o último trimestre, então diferenças são esperadas. "
               "Bancos e seguradoras usam um plano de contas próprio e podem ficar em branco. "
               "Confira sempre no site de relações com investidores da empresa.")


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

    aba_res, aba_sobre, aba_graf, aba_fund, aba_tend, aba_mom = st.tabs(
        ["Resumo", "Sobre o ativo", "Gráficos", "Fundamentos", "Tendência e médias",
         "Momento de compra" if MODO_PESSOAL else "Leitura técnica"])

    with aba_res:
        a, b, c = st.columns(3)
        rotulo_fund = "Fundamentos" if MODO_PESSOAL else "Critérios de referência"
        if fund and MODO_PESSOAL:
            a.metric(rotulo_fund, fund["veredito"], f"{fund['aprovados']}/{fund['total']} critérios",
                     delta_color="off")
        elif fund:
            a.metric(rotulo_fund, f"{fund['aprovados']} de {fund['total']}", "atendidos", delta_color="off")
        else:
            a.metric(rotulo_fund, "Não se aplica", "criptomoeda", delta_color="off")
        b.metric("Tendência", tendencia["geral"].replace("Tendência de ", "").replace("Tendência ", ""))
        if MODO_PESSOAL:
            c.metric("Momento técnico", momento["veredito"],
                     f"{momento['total']:+d} de ±{momento['maximo']}", delta_color="off")
            st.write(
                "O resumo cruza três leituras independentes: **valor** (fundamentos), "
                "**direção** (tendência) e **timing** (sinais técnicos). "
                "Um ativo pode ter bons fundamentos e mau momento técnico, e vice-versa. "
                "Veja o detalhe nas outras abas.")
        else:
            c.metric("Sinais técnicos", momento["curto"], "alta · baixa · neutro", delta_color="off")
            st.write(
                "O resumo cruza três leituras independentes: **valor** (fundamentos), "
                "**direção** (tendência) e **timing** (sinais técnicos). "
                "Os indicadores de valor e os sinais técnicos podem apontar em direções diferentes; "
                "cabe a você pesar cada leitura. "
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

    with aba_sobre:
        render_sobre(escolha, info, cripto, simbolo)

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
            st.markdown(f"### {fund['leitura']}")
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
            if ticker.endswith(".SA"):
                with st.expander("🏛️ Conferir com dados oficiais da CVM"):
                    if st.checkbox("Carregar os dados da CVM (pode levar alguns segundos)",
                                   key=f"cvm_{ticker}"):
                        with st.spinner("Baixando as demonstrações da CVM..."):
                            cv = cvm_fundamentos(ticker)
                        render_cvm(cv, info)

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
        if MODO_PESSOAL:
            st.markdown(f"### Momento técnico: {momento['veredito']} "
                        f"({momento['total']:+d} de ±{momento['maximo']})")
            sinais_df = pd.DataFrame([{"Sinal": n, "Pontos": f"{p:+d}", "Leitura": t}
                                      for n, p, t in momento["sinais"]])
        else:
            st.markdown(f"### Leitura técnica: {momento['leitura']}")
            sinais_df = pd.DataFrame(
                [{"Sinal": n, "Direção": "↑ alta" if p > 0 else ("↓ baixa" if p < 0 else "→ neutro"),
                  "Leitura": t} for n, p, t in momento["sinais"]])
        st.dataframe(sinais_df, hide_index=True, use_container_width=True)
        st.markdown("#### Níveis para observar" if MODO_PESSOAL else "#### Níveis de referência")
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Suporte (60 dias)", f"{simbolo} {br(momento['suporte'])}")
        n2.metric("Resistência (60 dias)", f"{simbolo} {br(momento['resistencia'])}")
        n3.metric("Banda inferior Bollinger", f"{simbolo} {br(momento['bb_inf'])}")
        n4.metric("Média 50 dias", f"{simbolo} {br(momento['sma50'])}")
        if MODO_PESSOAL:
            st.write(
                "Regiões próximas ao suporte, à banda inferior ou à média de 50 dias costumam ser "
                "pontos de observação para entradas, ao passo que rompimentos abaixo do suporte "
                "indicam que a leitura pode estar errada.")
            st.warning(
                "O placar mede apenas o momento técnico, com regras simples e conhecidas. "
                "Ele não prevê preços e ignora notícias, resultados e o seu perfil de risco. "
                "Não é recomendação de investimento.")
        else:
            st.write(
                "Suporte, resistência, bandas e médias são regiões onde o preço já reagiu no passado e "
                "que analistas usam como referência. Se o preço romper esses níveis, a leitura pode mudar.")
            st.warning(
                "Os sinais descrevem o comportamento recente do preço com regras simples e conhecidas. "
                "Eles não preveem preços e ignoram notícias, resultados e o seu perfil de risco. "
                "Conteúdo educacional; não é recomendação de investimento.")


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
            ("Momento técnico" if MODO_PESSOAL else "Sinais técnicos"): r["momento"] if r else "—"})
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

    st.session_state["_escolha"] = escolha
    t_an, t_fav, t_cmp, t_ap, t_ca, t_rk, t_b3, t_us, t_mc, t_al, t_nt, t_cr = st.tabs(
        ["Análise", "⭐ Favoritos", "⚖️ Comparador", "📚 Aprender", "💼 Carteira", "🛡️ Risco",
         "📊 B3 hoje", "🇺🇸 EUA hoje", "🪙 Mercado cripto", "🔔 Alertas", "📰 Notícias",
         "📰 Notícias cripto"])
    with t_an:
        if escolha:
            render_analise(escolha, periodo)
        else:
            st.info("Digite o nome ou código de um ativo na barra lateral.")
    with t_fav:
        render_favoritos(dados)
    with t_cmp:
        render_comparador(dados, escolha)
    with t_ap:
        render_aprender(dados)
    with t_ca:
        render_carteira(dados, escolha)
    with t_rk:
        render_risco(dados, escolha)
    with t_b3:
        render_b3_hoje(dados)
    with t_us:
        render_eua_hoje(dados)
    with t_mc:
        render_mercado_cripto()
    with t_al:
        render_alertas(dados, escolha, persistente)
    with t_nt:
        render_noticias(dados, escolha)
    with t_cr:
        render_cripto()
    st.divider()
    st.caption("Conteúdo educacional e informativo. Não constitui recomendação de investimento, "
               "oferta ou consultoria. Cada investidor é responsável por suas decisões. "
               "Dados de fontes públicas, que podem conter atrasos ou erros.")


if __name__ == "__main__":
    main()
