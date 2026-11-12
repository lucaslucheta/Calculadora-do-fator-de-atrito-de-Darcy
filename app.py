"""
Calculadora web do fator de atrito de Darcy - Diagrama de Moody.

Python 3.11+
Instalacao:
    python -m pip install -r requirements.txt
Execucao:
    python -m streamlit run app.py

A aplicacao abre uma interface no navegador, executada localmente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.optimize import brentq


# -----------------------------------------------------------------------------
# CONFIGURACOES FISICAS E NUMERICAS
# -----------------------------------------------------------------------------

RE_LAMINAR_MAX = 2300.0
RE_TRANSITION_MAX = 4000.0

# Intervalo convencional solicitado para a primeira tentativa do Brent.
F_MOODY_MIN = 0.008
F_MOODY_MAX = 0.10

# Intervalo expandido para nao limitar artificialmente as solucoes possiveis.
F_SEARCH_MIN = 1e-12
F_SEARCH_MAX = 10.0

# Criterio de engenharia configuravel para considerar plena turbulencia:
# termo de Reynolds <= 1% do termo de rugosidade na equacao de Colebrook.
ROUGHNESS_RATIO_TOL = 0.01

UNITS_TO_METERS: dict[str, float] = {
    "mm": 1e-3,
    "cm": 1e-2,
    "m": 1.0,
    "pol": 0.0254,
}

UNIT_LABELS = {
    "mm": "Milímetros (mm)",
    "cm": "Centímetros (cm)",
    "m": "Metros (m)",
    "pol": "Polegadas (pol)",
}


@dataclass(frozen=True, slots=True)
class MoodyResult:
    """Resultado do calculo do fator de atrito de Darcy."""

    reynolds: float
    relative_roughness: float
    friction_factor: float
    regime: str
    roughness_zone: str
    method: str
    reynolds_roughness_ratio: float | None = None


# -----------------------------------------------------------------------------
# VALIDACAO E SOLUCAO DAS EQUACOES
# -----------------------------------------------------------------------------


def validate_inputs(reynolds: float, relative_roughness: float) -> None:
    """Valida Re e epsilon/D antes de iniciar os calculos."""
    if not np.isfinite(reynolds) or reynolds <= 0:
        raise ValueError("O número de Reynolds deve ser positivo e finito.")
    if (
        not np.isfinite(relative_roughness)
        or relative_roughness < 0
        or relative_roughness >= 1
    ):
        raise ValueError("A rugosidade relativa deve satisfazer 0 ≤ ε/D < 1.")


def solve_with_brent(residual: Callable[[float], float]) -> float:
    """Resolve uma equacao escalar com Brent, expandindo o intervalo se preciso.

    A primeira tentativa respeita f entre 0,008 e 0,10, faixa usual do
    Diagrama de Moody. Se nao houver mudanca de sinal, usa um intervalo mais
    amplo. Isso evita tratar os limites graficos como limites fisicos absolutos.
    """
    for lower, upper in (
        (F_MOODY_MIN, F_MOODY_MAX),
        (F_SEARCH_MIN, F_SEARCH_MAX),
    ):
        f_lower = float(residual(lower))
        f_upper = float(residual(upper))

        if f_lower == 0.0:
            return lower
        if f_upper == 0.0:
            return upper

        if (
            np.isfinite(f_lower)
            and np.isfinite(f_upper)
            and np.signbit(f_lower) != np.signbit(f_upper)
        ):
            return float(
                brentq(
                    residual,
                    lower,
                    upper,
                    xtol=1e-14,
                    rtol=1e-12,
                    maxiter=200,
                )
            )

    raise RuntimeError(
        "Não foi possível delimitar uma raiz para a equação. "
        "Verifique os valores informados."
    )


def prandtl_karman_residual(friction_factor: float, reynolds: float) -> float:
    r"""Residuo de Prandtl-Karman para tubo hidraulicamente liso.

    1/sqrt(f) = 2 log10(Re sqrt(f)) - 0,8
    """
    sqrt_f = np.sqrt(friction_factor)
    return float(
        1.0 / sqrt_f
        - 2.0 * (np.log10(reynolds) + 0.5 * np.log10(friction_factor))
        + 0.8
    )


def solve_prandtl_karman(reynolds: float) -> float:
    """Calcula f para tubo liso no regime turbulento."""
    return solve_with_brent(
        lambda f: prandtl_karman_residual(f, reynolds)
    )


def colebrook_residual(
    friction_factor: float,
    reynolds: float,
    relative_roughness: float,
) -> float:
    r"""Residuo da equacao de Colebrook-White.

    1/sqrt(f) = -2 log10(ε/(3,7D) + 2,51/(Re sqrt(f)))

    A formulacao preserva os termos separados para reduzir problemas
    numericos em Reynolds elevados.
    """
    sqrt_f = np.sqrt(friction_factor)
    roughness_term = relative_roughness / 3.7
    reynolds_term = 2.51 / reynolds / sqrt_f
    argument = roughness_term + reynolds_term

    if argument <= 0.0 or not np.isfinite(argument):
        return float("nan")

    return float(1.0 / sqrt_f + 2.0 * np.log10(argument))


def solve_colebrook(reynolds: float, relative_roughness: float) -> float:
    """Resolve Colebrook-White com o metodo de Brent."""
    return solve_with_brent(
        lambda f: colebrook_residual(f, reynolds, relative_roughness)
    )


def calculate_term_ratio(
    reynolds: float,
    relative_roughness: float,
    friction_factor: float,
) -> tuple[float, float]:
    """Retorna log10(B/A) e B/A, termos de Reynolds/rugosidade de Colebrook.

    A = (ε/D)/3,7
    B = 2,51/(Re sqrt(f))
    """
    log_ratio = (
        np.log10(2.51)
        + np.log10(3.7)
        - np.log10(reynolds)
        - 0.5 * np.log10(friction_factor)
        - np.log10(relative_roughness)
    )

    if log_ratio > 308:
        ratio = float("inf")
    elif log_ratio < -323:
        ratio = 0.0
    else:
        ratio = float(10.0**log_ratio)

    return float(log_ratio), ratio


def solve_fully_rough(relative_roughness: float) -> float:
    r"""Equacao assintotica para tubo totalmente rugoso.

    1/sqrt(f) = -2 log10((ε/D)/3,7)
    """
    inverse_sqrt_f = -2.0 * (
        np.log10(relative_roughness) - np.log10(3.7)
    )
    if inverse_sqrt_f <= 0.0:
        raise ValueError("A rugosidade relativa não permite a solução assintótica.")
    return float(1.0 / inverse_sqrt_f**2)


def transition_interpolation(reynolds: float, relative_roughness: float) -> float:
    """Interpola suavemente f entre as solucoes laminar e turbulenta.

    A regiao 2300 <= Re <= 4000 e instavel e nao possui uma correlacao
    universal. Usa-se smoothstep w=3x²-2x³ para uma estimativa continua:

        f = (1-w) (64/Re) + w f_Colebrook

    Essa interpolacao e uma aproximacao computacional, nao uma lei fisica
    universal nem uma previsao de transicao para qualquer instalacao.
    """
    x = (reynolds - RE_LAMINAR_MAX) / (RE_TRANSITION_MAX - RE_LAMINAR_MAX)
    x = float(np.clip(x, 0.0, 1.0))
    weight = 3.0 * x**2 - 2.0 * x**3

    f_laminar = 64.0 / reynolds
    f_turbulent = solve_colebrook(reynolds, relative_roughness)
    return float((1.0 - weight) * f_laminar + weight * f_turbulent)


def calculate_darcy_friction_factor(
    reynolds: float,
    relative_roughness: float,
) -> MoodyResult:
    """Calcula o fator de Darcy e classifica o regime e a zona de rugosidade."""
    validate_inputs(reynolds, relative_roughness)

    if reynolds < RE_LAMINAR_MAX:
        return MoodyResult(
            reynolds=reynolds,
            relative_roughness=relative_roughness,
            friction_factor=64.0 / reynolds,
            regime="Laminar",
            roughness_zone="Não influencia f pela lei laminar",
            method="Hagen–Poiseuille: f = 64/Re",
        )

    if reynolds <= RE_TRANSITION_MAX:
        return MoodyResult(
            reynolds=reynolds,
            relative_roughness=relative_roughness,
            friction_factor=transition_interpolation(reynolds, relative_roughness),
            regime="Transição",
            roughness_zone="Não classificável com segurança nesta região instável",
            method="Interpolação smoothstep entre 64/Re e Colebrook–White",
        )

    if relative_roughness == 0.0:
        return MoodyResult(
            reynolds=reynolds,
            relative_roughness=relative_roughness,
            friction_factor=solve_prandtl_karman(reynolds),
            regime="Turbulento",
            roughness_zone="Tubo liso (ε/D = 0)",
            method="Prandtl–Kármán, resolvida numericamente com Brent",
        )

    f_colebrook = solve_colebrook(reynolds, relative_roughness)
    log_ratio, ratio = calculate_term_ratio(
        reynolds, relative_roughness, f_colebrook
    )

    if log_ratio <= np.log10(ROUGHNESS_RATIO_TOL):
        return MoodyResult(
            reynolds=reynolds,
            relative_roughness=relative_roughness,
            friction_factor=solve_fully_rough(relative_roughness),
            regime="Turbulento",
            roughness_zone="Totalmente rugoso / plena turbulência",
            method="Equação assintótica de Colebrook–White, independente de Re",
            reynolds_roughness_ratio=ratio,
        )

    return MoodyResult(
        reynolds=reynolds,
        relative_roughness=relative_roughness,
        friction_factor=f_colebrook,
        regime="Turbulento",
        roughness_zone="Transição turbulenta",
        method="Colebrook–White, resolvida numericamente com Brent",
        reynolds_roughness_ratio=ratio,
    )


# -----------------------------------------------------------------------------
# GRAFICO DO DIAGRAMA DE MOODY
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def build_moody_curves() -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Gera curvas de Colebrook para rugosidades relativas usuais.

    O cache evita recalcular as curvas a cada interacao com a interface.
    """
    reynolds_values = np.geomspace(4_001.0, 1e8, 75)
    roughness_values = (
        0.0,
        1e-5,
        5e-5,
        1e-4,
        2e-4,
        5e-4,
        1e-3,
        2e-3,
        5e-3,
        1e-2,
        2e-2,
        5e-2,
    )

    curves: dict[float, np.ndarray] = {}
    for relative_roughness in roughness_values:
        if relative_roughness == 0.0:
            values = [solve_prandtl_karman(float(re)) for re in reynolds_values]
        else:
            values = [
                solve_colebrook(float(re), relative_roughness)
                for re in reynolds_values
            ]
        curves[relative_roughness] = np.asarray(values, dtype=float)

    return reynolds_values, curves


def make_moody_figure(selected: MoodyResult | None) -> plt.Figure:
    """Monta o grafico de Moody e, se houver, destaca o ponto calculado."""
    re_values, curves = build_moody_curves()
    fig, ax = plt.subplots(figsize=(11, 6.4), constrained_layout=True)

    # Linha analitica laminar.
    re_laminar = np.geomspace(500.0, RE_LAMINAR_MAX, 100)
    ax.plot(re_laminar, 64.0 / re_laminar, color="black", linewidth=2.0,
            label="Laminar: f = 64/Re")

    # Faixa de transicao destacada.
    ax.axvspan(RE_LAMINAR_MAX, RE_TRANSITION_MAX, alpha=0.12,
               label="Região de transição (estimativa)")

    # Curvas turbulentas para rugosidades relativas padrao.
    for relative_roughness, friction_values in curves.items():
        label = (
            "ε/D = 0 (liso)"
            if relative_roughness == 0.0
            else f"ε/D = {relative_roughness:g}"
        )
        ax.plot(re_values, friction_values, linewidth=1.25, label=label)

    if selected is not None:
        ax.scatter(
            [selected.reynolds],
            [selected.friction_factor],
            s=95,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            zorder=10,
            label="Ponto calculado",
        )
        ax.annotate(
            f"Re = {selected.reynolds:.4g}\nf = {selected.friction_factor:.5g}",
            (selected.reynolds, selected.friction_factor),
            xytext=(9, 10),
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "alpha": 0.85},
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(500, 1e8)
    ax.set_ylim(0.005, 0.2)
    ax.set_xlabel("Número de Reynolds (Re)")
    ax.set_ylabel("Fator de atrito de Darcy (f)")
    ax.set_title("Diagrama de Moody — curvas calculadas por Prandtl–Kármán e Colebrook–White")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    return fig


# -----------------------------------------------------------------------------
# INTERFACE STREAMLIT
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Calculadora do Diagrama de Moody",
    page_icon="📐",
    layout="wide",
)

st.markdown(
    """
    <style>
      .main .block-container {padding-top: 2rem; padding-bottom: 2.5rem;}
      div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 0.8rem 1rem;
        border-radius: 0.7rem;
      }
      .small-note {font-size: 0.88rem; opacity: 0.82;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📐 Calculadora do fator de atrito de Darcy")
st.caption("Diagrama de Moody • Prandtl–Kármán • Colebrook–White • Método de Brent")
st.write(
    "Informe o número de Reynolds e a rugosidade do tubo. "
    "A aplicação calcula o fator de atrito de Darcy e identifica o regime de escoamento."
)

if "moody_result" not in st.session_state:
    st.session_state.moody_result = None
if "moody_error" not in st.session_state:
    st.session_state.moody_error = None

input_col, result_col = st.columns([0.95, 1.05], gap="large")

with input_col:
    st.subheader("1. Dados de entrada")

    with st.form("moody_calculation_form"):
        reynolds_input = st.number_input(
            "Número de Reynolds (Re)",
            min_value=0.000001,
            max_value=1e15,
            value=100000.0,
            step=1000.0,
            format="%.6g",
            help="Número de Reynolds adimensional e estritamente positivo.",
        )

        roughness_mode = st.radio(
            "Como deseja informar a rugosidade?",
            ("Rugosidade relativa (ε/D)", "Rugosidade absoluta (ε) e diâmetro (D)"),
            help="As unidades de ε e D podem ser selecionadas independentemente.",
        )

        relative_roughness_input: float | None = None
        epsilon_input: float | None = None
        epsilon_unit_input: str | None = None
        diameter_input: float | None = None
        diameter_unit_input: str | None = None

        if roughness_mode == "Rugosidade relativa (ε/D)":
            relative_roughness_input = st.number_input(
                "Rugosidade relativa (ε/D)",
                min_value=0.0,
                max_value=0.999999999,
                value=0.001,
                format="%.9g",
                help="Use zero para representar um tubo hidraulicamente liso.",
            )
        else:
            epsilon_col, epsilon_unit_col = st.columns([1.4, 1])
            with epsilon_col:
                epsilon_input = st.number_input(
                    "Rugosidade absoluta (ε)",
                    min_value=0.0,
                    value=0.1,
                    format="%.8g",
                )
            with epsilon_unit_col:
                epsilon_unit_input = st.selectbox(
                    "Unidade de ε",
                    options=tuple(UNITS_TO_METERS.keys()),
                    format_func=lambda unit: UNIT_LABELS[unit],
                    index=0,
                )

            diameter_col, diameter_unit_col = st.columns([1.4, 1])
            with diameter_col:
                diameter_input = st.number_input(
                    "Diâmetro interno (D)",
                    min_value=0.000001,
                    value=100.0,
                    format="%.8g",
                )
            with diameter_unit_col:
                diameter_unit_input = st.selectbox(
                    "Unidade de D",
                    options=tuple(UNITS_TO_METERS.keys()),
                    format_func=lambda unit: UNIT_LABELS[unit],
                    index=0,
                )

        submitted = st.form_submit_button(
            "Calcular fator de atrito", type="primary", use_container_width=True
        )

    if submitted:
        try:
            if roughness_mode == "Rugosidade relativa (ε/D)":
                assert relative_roughness_input is not None
                relative_roughness = float(relative_roughness_input)
            else:
                assert epsilon_input is not None
                assert epsilon_unit_input is not None
                assert diameter_input is not None
                assert diameter_unit_input is not None

                epsilon_m = epsilon_input * UNITS_TO_METERS[epsilon_unit_input]
                diameter_m = diameter_input * UNITS_TO_METERS[diameter_unit_input]
                if diameter_m <= 0.0:
                    raise ValueError("O diâmetro deve ser maior que zero.")
                relative_roughness = epsilon_m / diameter_m

            result = calculate_darcy_friction_factor(
                float(reynolds_input), float(relative_roughness)
            )
            st.session_state.moody_result = result
            st.session_state.moody_error = None
        except (ValueError, RuntimeError, ArithmeticError, AssertionError) as error:
            st.session_state.moody_result = None
            st.session_state.moody_error = str(error) or "Verifique os dados informados."

    if st.session_state.moody_error:
        st.error(st.session_state.moody_error)

with result_col:
    st.subheader("2. Resultado")
    result: MoodyResult | None = st.session_state.moody_result

    if result is None:
        st.info(
            "Preencha os dados ao lado e clique em **Calcular fator de atrito** "
            "para visualizar o resultado. Os valores iniciais servem como exemplo."
        )
    else:
        st.metric(
            "Fator de atrito de Darcy (f)",
            f"{result.friction_factor:.8f}",
        )
        metric_col_1, metric_col_2 = st.columns(2)
        metric_col_1.metric("Regime de escoamento", result.regime)
        metric_col_2.metric("Número de Reynolds", f"{result.reynolds:.5g}")

        st.markdown("**Classificação da rugosidade**")
        st.write(result.roughness_zone)
        st.markdown("**Método utilizado**")
        st.write(result.method)
        st.markdown("**Rugosidade relativa calculada**")
        st.code(f"ε/D = {result.relative_roughness:.10g}")

        if result.reynolds_roughness_ratio is not None:
            ratio = result.reynolds_roughness_ratio
            ratio_text = f"{ratio:.6g}" if np.isfinite(ratio) else "muito elevado"
            st.caption(
                "Razão B/A (termo de Reynolds / termo de rugosidade): "
                f"{ratio_text}. Critério adotado para plena turbulência: B/A ≤ "
                f"{ROUGHNESS_RATIO_TOL:.0%}."
            )

st.divider()
st.subheader("3. Diagrama de Moody")
st.caption(
    "As curvas turbulentas são calculadas numericamente. A faixa sombreada "
    "indica a região de transição; o símbolo ★ marca o ponto calculado."
)

with st.spinner("Preparando as curvas do Diagrama de Moody..."):
    moody_figure = make_moody_figure(st.session_state.moody_result)
st.pyplot(moody_figure, use_container_width=True)
plt.close(moody_figure)

with st.expander("Sobre os modelos e as hipóteses"):
    st.markdown(
        r"""
        **Regime laminar**

        \[f = \frac{64}{Re}\]

        **Tubo liso em regime turbulento — Prandtl–Kármán**

        \[\frac{1}{\sqrt{f}} = 2\log_{10}(Re\sqrt{f}) - 0,8\]

        **Tubos rugosos — Colebrook–White**

        \[\frac{1}{\sqrt{f}} = -2\log_{10}\left(\frac{\varepsilon/D}{3,7} + \frac{2,51}{Re\sqrt{f}}\right)\]

        **Região totalmente rugosa — limite assintótico**

        \[f = \frac{1}{\left[-2\log_{10}\left((\varepsilon/D)/3,7\right)\right]^2}\]

        **Atenção:** entre Re = 2300 e Re = 4000, o aplicativo apresenta uma
        interpolação suave entre a expressão laminar e Colebrook–White. Essa
        faixa é fisicamente instável; a interpolação é uma estimativa prática,
        não uma correlação universal. A identificação de plena turbulência usa
        o critério explícito B/A ≤ 1%, configurável no código.

        O fator apresentado é o **fator de Darcy**. Para obter o fator de
        Fanning, divida o resultado por quatro.
        """
    )

st.caption("Aplicação local em Python • Streamlit • NumPy • SciPy • Matplotlib")
