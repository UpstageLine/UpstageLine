from pathlib import Path
import math

import flet as ft
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import chi2_contingency
except Exception:
    chi2_contingency = None


# ============================================================
# ПУТИ И НАСТРОЙКИ
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
OUTPUTS_DIR = BASE_DIR / "outputs"

DATA_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

FLET_READY_PATH = DATA_DIR / "flet_ready_diabetes_data.csv"
RAW_DATA_PATH = DATA_DIR / "diabetic_data.csv"

APP_WIDTH = 1220
ALPHA = 0.05


# ============================================================
# ЦВЕТА И ПОДПИСИ
# ============================================================

COLORS = {
    "page": "#0B1120",
    "card": "#111827",
    "card_2": "#1F2937",
    "card_3": "#172554",
    "header": "#020617",
    "table_header": "#243244",
    "table_row": "#111827",
    "table_row_alt": "#182233",
    "white": "#FFFFFF",
    "text": "#D6DCE8",
    "muted": "#AAB2C0",
    "muted_2": "#7F8EA3",
    "green": "#86EFAC",
    "green_bright": "#22C55E",
    "blue": "#60A5FA",
    "yellow": "#FACC15",
    "red": "#FCA5A5",
    "purple": "#A78BFA",
    "active_nav": "#2563EB",
    "inactive_nav": "#1F2937",
}

FACTOR_LABELS = {
    "a1c_high_8": "HbA1c > 8%",
    "diagnoses_3plus": "3 и более диагноза",
    "older_70": "Возраст старше 70 лет",
    "had_previous_inpatient": "Предыдущие стационарные обращения",
    "had_emergency_visits": "Экстренные обращения",
    "had_outpatient_visits": "Амбулаторные обращения",
    "many_medications": "Высокая лекарственная нагрузка",
    "insulin_used": "Использование инсулина",
    "treatment_changed": "Изменение лечения",
    "glucose_high": "Повышенная глюкоза",
    "cardiovascular_disease": "Сердечно-сосудистый диагноз",
    "severe_diabetes": "Учебный индикатор тяжёлого течения",
    "gender": "Пол пациента",
    "race": "Расовая группа",
}

RISK_FEATURES = [
    "diagnoses_3plus",
    "older_70",
    "had_previous_inpatient",
    "had_emergency_visits",
    "many_medications",
    "insulin_used",
]

CANDIDATE_RISK_FACTORS = [
    "had_previous_inpatient",
    "had_emergency_visits",
    "had_outpatient_visits",
    "older_70",
    "diagnoses_3plus",
    "many_medications",
    "insulin_used",
    "treatment_changed",
    "glucose_high",
    "cardiovascular_disease",
    "severe_diabetes",
    "a1c_high_8",
]


# ============================================================
# СОВМЕСТИМЫЕ КОМПОНЕНТЫ FLET
# ============================================================

def create_button(label, handler):
    try:
        return ft.ElevatedButton(label, on_click=handler)
    except Exception:
        return ft.ElevatedButton(content=ft.Text(label), on_click=handler)


def create_text(value, size=14, color=None, weight=None):
    if color is None:
        color = COLORS["text"]

    kwargs = {
        "value": str(value),
        "size": size,
        "color": color,
    }

    if weight is not None:
        kwargs["weight"] = weight

    return ft.Text(**kwargs)


def create_container(
    content,
    bgcolor=None,
    padding=16,
    border_radius=12,
    width=None,
    height=None,
    on_click=None,
):
    if bgcolor is None:
        bgcolor = COLORS["card"]

    kwargs = {
        "content": content,
        "bgcolor": bgcolor,
        "padding": padding,
        "border_radius": border_radius,
    }

    if width is not None:
        kwargs["width"] = width

    if height is not None:
        kwargs["height"] = height

    if on_click is not None:
        kwargs["on_click"] = on_click

    return ft.Container(**kwargs)


def spacer(height=8):
    return ft.Container(height=height)


# ============================================================
# ЗАГРУЗКА И ПРЕДОБРАБОТКА
# ============================================================

def icd9_category(value):
    if pd.isna(value) or str(value).strip() == "":
        return "Не указан"

    text = str(value).strip()

    if text.startswith("V"):
        return "Дополнительные факторы"

    if text.startswith("E"):
        return "Внешние причины"

    try:
        number = float(text)
    except ValueError:
        return "Другие"

    if 250 <= number < 251:
        return "Диабет"
    if 390 <= number < 460 or number == 785:
        return "Сердечно-сосудистые"
    if 460 <= number < 520 or number == 786:
        return "Респираторные"
    if 520 <= number < 580 or number == 787:
        return "Пищеварительная система"
    if 580 <= number < 630 or number == 788:
        return "Мочеполовая система"
    if 140 <= number < 240:
        return "Новообразования"
    if 710 <= number < 740:
        return "Опорно-двигательная система"
    if 800 <= number < 1000:
        return "Травмы"

    return "Другие"


def read_csv_safely(path):
    """
    keep_default_na=False сохраняет строковое значение None в A1Cresult.
    Символ ? обрабатывается отдельно.
    """
    return pd.read_csv(
        path,
        low_memory=False,
        keep_default_na=False,
    )


def prepare_dataset(dataframe):
    """
    Приводит исходный или ранее подготовленный CSV к той же логике,
    которая используется в финальном Jupyter Notebook.
    """
    df = dataframe.copy()
    df = df.replace("?", np.nan)
    df = df.replace("", np.nan)

    required_base_columns = {
        "patient_nbr",
        "gender",
        "age",
        "number_diagnoses",
        "num_medications",
        "number_inpatient",
        "number_emergency",
        "number_outpatient",
        "A1Cresult",
        "max_glu_serum",
        "insulin",
        "change",
        "readmitted",
    }

    missing_base = sorted(required_base_columns - set(df.columns))

    if missing_base:
        raise ValueError(
            "В выбранном CSV отсутствуют обязательные столбцы: "
            + ", ".join(missing_base)
        )

    # Фильтрация выполняется только тогда, когда соответствующие поля есть.
    if "gender" in df.columns:
        df = df[df["gender"] != "Unknown/Invalid"].copy()

    if "discharge_disposition_id" in df.columns:
        death_or_hospice_ids = [11, 13, 14, 19, 20, 21]
        df = df[
            ~pd.to_numeric(
                df["discharge_disposition_id"],
                errors="coerce",
            ).isin(death_or_hospice_ids)
        ].copy()

    # Для анализа используется первый доступный эпизод каждого пациента.
    if "encounter_id" in df.columns:
        df["encounter_id"] = pd.to_numeric(
            df["encounter_id"],
            errors="coerce",
        )
        df = df.sort_values("encounter_id")

    df = df.drop_duplicates(
        subset="patient_nbr",
        keep="first",
    ).copy()

    numeric_columns = [
        "age_mid",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df["readmitted_30"] = (
        df["readmitted"].astype(str) == "<30"
    ).astype(int)

    age_mapping = {
        "[0-10)": 5,
        "[10-20)": 15,
        "[20-30)": 25,
        "[30-40)": 35,
        "[40-50)": 45,
        "[50-60)": 55,
        "[60-70)": 65,
        "[70-80)": 75,
        "[80-90)": 85,
        "[90-100)": 95,
    }

    df["age_mid"] = df["age"].map(age_mapping)
    df["age_group"] = pd.cut(
        df["age_mid"],
        bins=[0, 30, 50, 70, 100],
        labels=[
            "до 30 лет",
            "30–50 лет",
            "50–70 лет",
            "старше 70 лет",
        ],
        include_lowest=True,
    )
    df["older_70"] = (df["age_mid"] > 70).astype(int)

    a1c_mapping = {
        "Norm": 0,
        ">7": 1,
        ">8": 2,
    }

    df["a1c_measured"] = (
        df["A1Cresult"].isin(["Norm", ">7", ">8"])
    ).astype(int)
    df["a1c_high_8"] = (
        df["A1Cresult"] == ">8"
    ).astype(int)
    df["a1c_level"] = df["A1Cresult"].map(a1c_mapping)
    df["glucose_high"] = (
        df["max_glu_serum"].isin([">200", ">300"])
    ).astype(int)

    for diagnosis_column in ["diag_1", "diag_2", "diag_3"]:
        if diagnosis_column in df.columns:
            df[f"{diagnosis_column}_category"] = (
                df[diagnosis_column].apply(icd9_category)
            )

    diagnosis_category_columns = [
        column
        for column in [
            "diag_1_category",
            "diag_2_category",
            "diag_3_category",
        ]
        if column in df.columns
    ]

    if diagnosis_category_columns:
        df["cardiovascular_disease"] = (
            df[diagnosis_category_columns]
            .eq("Сердечно-сосудистые")
            .any(axis=1)
            .astype(int)
        )
    else:
        df["cardiovascular_disease"] = 0

    df["diagnosis_count"] = df["number_diagnoses"]
    df["diagnoses_3plus"] = (
        df["number_diagnoses"] >= 3
    ).astype(int)
    df["diagnosis_group"] = pd.cut(
        df["number_diagnoses"],
        bins=[-1, 2, 5, np.inf],
        labels=["0–2", "3–5", "6 и более"],
    )

    medication_columns = [
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "examide",
        "citoglipton",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]

    medication_columns = [
        column
        for column in medication_columns
        if column in df.columns
    ]

    medication_threshold = float(
        df["num_medications"].median()
    )

    df["medications_count"] = df["num_medications"]

    if medication_columns:
        df["active_diabetes_medications_count"] = (
            df[medication_columns]
            .ne("No")
            .sum(axis=1)
        )
    else:
        df["active_diabetes_medications_count"] = 0

    df["insulin_used"] = (
        df["insulin"] != "No"
    ).astype(int)
    df["treatment_changed"] = (
        df["change"] == "Ch"
    ).astype(int)
    df["many_medications"] = (
        df["num_medications"] >= medication_threshold
    ).astype(int)
    df["had_previous_inpatient"] = (
        df["number_inpatient"] > 0
    ).astype(int)
    df["had_emergency_visits"] = (
        df["number_emergency"] > 0
    ).astype(int)
    df["had_outpatient_visits"] = (
        df["number_outpatient"] > 0
    ).astype(int)

    df["severe_diabetes"] = (
        (df["a1c_high_8"] == 1)
        & (
            (df["cardiovascular_disease"] == 1)
            | (df["number_diagnoses"] >= 6)
            | (df["insulin_used"] == 1)
        )
    ).astype(int)

    # HbA1c > 8% намеренно не включается:
    # обязательная гипотеза о повышении риска не подтвердилась.
    df["risk_score"] = df[RISK_FEATURES].sum(axis=1)
    df["risk_group"] = pd.cut(
        df["risk_score"],
        bins=[-1, 1, 3, len(RISK_FEATURES)],
        labels=[
            "низкий риск",
            "средний риск",
            "высокий риск",
        ],
    )

    return df


def load_dataset(path=None):
    selected_path = Path(path) if path else FLET_READY_PATH

    if not selected_path.exists():
        if RAW_DATA_PATH.exists():
            selected_path = RAW_DATA_PATH
        else:
            raise FileNotFoundError(
                "Не найден подготовленный или исходный CSV "
                "в папке 04_Flet_app/data."
            )

    raw_df = read_csv_safely(selected_path)
    prepared_df = prepare_dataset(raw_df)

    return prepared_df, selected_path


# ============================================================
# СТАТИСТИКА
# ============================================================

def cramers_v(contingency_table):
    if chi2_contingency is None:
        return np.nan

    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return np.nan

    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.to_numpy().sum()
    denominator = min(
        contingency_table.shape[0] - 1,
        contingency_table.shape[1] - 1,
    )

    if n == 0 or denominator <= 0:
        return np.nan

    return math.sqrt(chi2 / (n * denominator))


def safe_chi_square(table):
    if chi2_contingency is None:
        raise ImportError(
            "Библиотека scipy не установлена. "
            "Выполни: python -m pip install scipy"
        )

    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError(
            "Недостаточно категорий для статистического теста."
        )

    return chi2_contingency(table)


def binary_hypothesis(data, factor, number, hypothesis_text):
    subset = data[
        [factor, "readmitted_30"]
    ].dropna().copy()

    table = pd.crosstab(
        subset[factor],
        subset["readmitted_30"],
    )

    if 0 not in table.index or 1 not in table.index:
        return {
            "№": number,
            "Гипотеза": hypothesis_text,
            "Фактор": FACTOR_LABELS.get(factor, factor),
            "N": len(subset),
            "Риск без фактора, %": np.nan,
            "Риск с фактором, %": np.nan,
            "Разница, п.п.": np.nan,
            "Относительный риск": np.nan,
            "Odds Ratio": np.nan,
            "p-value": np.nan,
            "Cramer's V": np.nan,
            "Итог": "недостаточно данных",
        }

    chi2, p_value, _, _ = safe_chi_square(table)

    rates = (
        subset
        .groupby(factor)["readmitted_30"]
        .agg(["count", "mean"])
    )

    rate_0 = rates.loc[0, "mean"]
    rate_1 = rates.loc[1, "mean"]

    relative_risk = (
        rate_1 / rate_0
        if rate_0 > 0
        else np.nan
    )

    a = table.loc[1, 1] + 0.5
    b = table.loc[1, 0] + 0.5
    c = table.loc[0, 1] + 0.5
    d = table.loc[0, 0] + 0.5

    odds_ratio = (a / b) / (c / d)

    direction_is_expected = rate_1 > rate_0
    confirmed = (
        p_value < ALPHA
        and direction_is_expected
    )

    return {
        "№": number,
        "Гипотеза": hypothesis_text,
        "Фактор": FACTOR_LABELS.get(factor, factor),
        "N": len(subset),
        "Риск без фактора, %": rate_0 * 100,
        "Риск с фактором, %": rate_1 * 100,
        "Разница, п.п.": (rate_1 - rate_0) * 100,
        "Относительный риск": relative_risk,
        "Odds Ratio": odds_ratio,
        "p-value": p_value,
        "Cramer's V": cramers_v(table),
        "Итог": (
            "подтверждается"
            if confirmed
            else "не подтверждается"
        ),
    }


def categorical_hypothesis(data, factor, number, hypothesis_text):
    subset = data[
        [factor, "readmitted_30"]
    ].dropna().copy()

    table = pd.crosstab(
        subset[factor],
        subset["readmitted_30"],
    )

    chi2, p_value, _, _ = safe_chi_square(table)

    rates = (
        subset
        .groupby(factor)["readmitted_30"]
        .agg(["count", "mean"])
    )

    effect = cramers_v(table)

    if p_value >= ALPHA:
        result_text = "не подтверждается"
    elif effect < 0.1:
        result_text = "частично подтверждается: связь слабая"
    else:
        result_text = "подтверждается"

    return {
        "№": number,
        "Гипотеза": hypothesis_text,
        "Фактор": FACTOR_LABELS.get(factor, factor),
        "N": len(subset),
        "Риск без фактора, %": np.nan,
        "Риск с фактором, %": np.nan,
        "Разница, п.п.": (
            rates["mean"].max()
            - rates["mean"].min()
        ) * 100,
        "Относительный риск": np.nan,
        "Odds Ratio": np.nan,
        "p-value": p_value,
        "Cramer's V": effect,
        "Итог": result_text,
    }


def calculate_hypotheses(df):
    rows = []

    h1_data = df[df["a1c_measured"] == 1].copy()
    rows.append(
        binary_hypothesis(
            h1_data,
            "a1c_high_8",
            1,
            "HbA1c > 8% повышает риск readmitted <30",
        )
    )

    rows.append(
        binary_hypothesis(
            df,
            "diagnoses_3plus",
            2,
            "Три и более диагноза повышают риск readmitted <30",
        )
    )

    rows.append(
        binary_hypothesis(
            df,
            "older_70",
            3,
            "Возраст старше 70 лет повышает риск readmitted <30",
        )
    )

    gender_data = df[
        df["gender"].isin(["Male", "Female"])
    ].copy()

    rows.append(
        categorical_hypothesis(
            gender_data,
            "gender",
            4,
            "Мужчины и женщины имеют разный риск readmitted <30",
        )
    )

    race_data = df[df["race"].notna()].copy()

    rows.append(
        categorical_hypothesis(
            race_data,
            "race",
            5,
            "Раса связана с вероятностью readmitted <30",
        )
    )

    return pd.DataFrame(rows)


def calculate_binary_factor_risk(data, factor):
    result = binary_hypothesis(
        data,
        factor,
        0,
        FACTOR_LABELS.get(factor, factor),
    )

    return {
        "Фактор": FACTOR_LABELS.get(factor, factor),
        "N": result["N"],
        "Риск без фактора, %": result["Риск без фактора, %"],
        "Риск с фактором, %": result["Риск с фактором, %"],
        "Разница, п.п.": result["Разница, п.п."],
        "Относительный риск": result["Относительный риск"],
        "Odds Ratio": result["Odds Ratio"],
        "p-value": result["p-value"],
        "Cramer's V": result["Cramer's V"],
    }


def calculate_risk_factors(df):
    rows = []

    for factor in CANDIDATE_RISK_FACTORS:
        if factor in df.columns:
            rows.append(
                calculate_binary_factor_risk(
                    df,
                    factor,
                )
            )

    result = pd.DataFrame(rows)

    return result.sort_values(
        "Разница, п.п.",
        ascending=False,
    )


# ============================================================
# ФОРМАТИРОВАНИЕ
# ============================================================

def format_percent(value):
    if pd.isna(value):
        return "—"
    return f"{value * 100:.2f}%"


def format_number(value):
    try:
        return f"{int(value):,}".replace(",", " ")
    except Exception:
        return str(value)


def format_float(value, digits=3):
    if pd.isna(value):
        return "—"
    return f"{value:.{digits}f}"


def format_p_value(value):
    if pd.isna(value):
        return "—"
    if value < 0.000001:
        return "< 0,000001"
    return f"{value:.6f}".replace(".", ",")


def short_text(value, max_len=62):
    text = str(value)

    if len(text) <= max_len:
        return text

    return text[:max_len - 3] + "..."


# ============================================================
# КАРТОЧКИ И ТАБЛИЦЫ
# ============================================================

def make_metric_card(title, value, subtitle="", explanation=""):
    controls = [
        create_text(
            title,
            size=13,
            color=COLORS["muted"],
        ),
        create_text(
            str(value),
            size=25,
            color=COLORS["white"],
            weight="bold",
        ),
    ]

    if subtitle:
        controls.append(
            create_text(
                subtitle,
                size=12,
                color=COLORS["green"],
            )
        )

    if explanation:
        controls.append(
            create_text(
                explanation,
                size=11,
                color=COLORS["muted_2"],
            )
        )

    return create_container(
        content=ft.Column(
            controls=controls,
            spacing=5,
        ),
        bgcolor=COLORS["card_2"],
        padding=16,
        border_radius=14,
        width=280,
    )


def make_text_block(title, body, accent=False):
    return create_container(
        content=ft.Column(
            controls=[
                create_text(
                    title,
                    size=19,
                    color=(
                        COLORS["green"]
                        if accent
                        else COLORS["white"]
                    ),
                    weight="bold",
                ),
                create_text(
                    body,
                    size=13,
                    color=COLORS["text"],
                ),
            ],
            spacing=8,
        ),
        bgcolor=(
            COLORS["card_3"]
            if accent
            else COLORS["card"]
        ),
        padding=18,
        border_radius=14,
        width=APP_WIDTH,
    )


def make_instruction_block():
    return create_container(
        content=ft.Column(
            controls=[
                create_text(
                    "Правила работы с интерфейсом",
                    size=21,
                    color=COLORS["white"],
                    weight="bold",
                ),
                create_text(
                    "1. Откройте раздел «Загрузка данных».\n"
                    "2. Можно использовать подготовленный CSV проекта или вставить путь к другому CSV-файлу.\n"
                    "3. «Общая сводка» показывает структуру выборки и основные графики.\n"
                    "4. «Проверка гипотез» выполняет пять обязательных статистических проверок.\n"
                    "5. «Факторы риска» ранжирует признаки по изменению доли readmitted <30.\n"
                    "6. «Примеры высокого риска» показывает записи с наибольшим учебным risk_score.",
                    size=14,
                    color=COLORS["white"],
                ),
            ],
            spacing=10,
        ),
        bgcolor="#1D4ED8",
        padding=20,
        border_radius=16,
        width=APP_WIDTH,
    )


def make_status_block(status_text_control):
    return create_container(
        content=ft.Column(
            controls=[
                create_text(
                    "Итог текущего раздела",
                    size=18,
                    color=COLORS["green"],
                    weight="bold",
                ),
                status_text_control,
            ],
            spacing=8,
        ),
        bgcolor="#052E16",
        padding=18,
        border_radius=14,
        width=APP_WIDTH,
    )


def make_cell(text, width, header=False):
    return create_container(
        content=create_text(
            short_text(text, 90),
            size=13 if header else 12,
            color=(
                COLORS["white"]
                if header
                else COLORS["text"]
            ),
            weight="bold" if header else None,
        ),
        bgcolor=(
            COLORS["table_header"]
            if header
            else COLORS["table_row"]
        ),
        padding=8,
        border_radius=6,
        width=width,
    )


def make_table(title, columns, rows, max_rows=20):
    header_row = ft.Row(
        controls=[
            make_cell(
                column["title"],
                column["width"],
                header=True,
            )
            for column in columns
        ],
        spacing=4,
    )

    table_rows = [header_row]

    for index, row in enumerate(rows[:max_rows]):
        row_controls = []

        for column in columns:
            value = row.get(column["key"], "")
            formatter = column.get("formatter")

            if formatter is not None:
                value = formatter(value)

            row_controls.append(
                create_container(
                    content=create_text(
                        short_text(
                            value,
                            column.get("max_len", 80),
                        ),
                        size=12,
                        color=COLORS["text"],
                    ),
                    bgcolor=(
                        COLORS["table_row_alt"]
                        if index % 2
                        else COLORS["table_row"]
                    ),
                    padding=8,
                    border_radius=6,
                    width=column["width"],
                )
            )

        table_rows.append(
            ft.Row(
                controls=row_controls,
                spacing=4,
            )
        )

    return create_container(
        content=ft.Column(
            controls=[
                create_text(
                    title,
                    size=19,
                    color=COLORS["white"],
                    weight="bold",
                ),
                ft.Column(
                    controls=table_rows,
                    spacing=4,
                ),
            ],
            spacing=8,
        ),
        bgcolor=COLORS["card"],
        padding=18,
        border_radius=14,
        width=APP_WIDTH,
    )


# ============================================================
# ГРАФИКИ
# ============================================================

def clean_old_charts():
    for image_path in ASSETS_DIR.glob("flet_*.png"):
        try:
            image_path.unlink()
        except OSError:
            pass


def apply_dark_chart_style(ax):
    figure = ax.figure
    figure.patch.set_facecolor(COLORS["card"])
    ax.set_facecolor(COLORS["card"])

    ax.title.set_color(COLORS["white"])
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.tick_params(axis="x", colors=COLORS["text"])
    ax.tick_params(axis="y", colors=COLORS["text"])

    for spine in ax.spines.values():
        spine.set_color("#334155")

    ax.grid(
        True,
        color="#334155",
        alpha=0.35,
    )


def save_chart(filename):
    path = ASSETS_DIR / filename

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        facecolor=COLORS["card"],
    )
    plt.close()

    return filename


def make_chart_card(title, image_name, explanation):
    return create_container(
        content=ft.Column(
            controls=[
                create_text(
                    title,
                    size=19,
                    color=COLORS["white"],
                    weight="bold",
                ),
                create_text(
                    explanation,
                    size=12,
                    color=COLORS["muted"],
                ),
                ft.Image(
                    src=image_name,
                    width=1040,
                ),
            ],
            spacing=10,
        ),
        bgcolor=COLORS["card"],
        padding=18,
        border_radius=14,
        width=APP_WIDTH,
    )


def create_age_chart(df):
    plt.rcParams["font.family"] = "DejaVu Sans"
    _, ax = plt.subplots(figsize=(10, 4.8))

    age_order = [
        "[0-10)",
        "[10-20)",
        "[20-30)",
        "[30-40)",
        "[40-50)",
        "[50-60)",
        "[60-70)",
        "[70-80)",
        "[80-90)",
        "[90-100)",
    ]

    counts = (
        df["age"]
        .value_counts()
        .reindex(age_order)
        .fillna(0)
    )

    ax.bar(
        counts.index,
        counts.values,
        color="#60A5FA",
    )
    ax.set_title(
        "Распределение уникальных пациентов по возрастным группам",
        pad=14,
    )
    ax.set_xlabel("Возрастная группа")
    ax.set_ylabel("Количество пациентов")
    ax.tick_params(
        axis="x",
        rotation=35,
    )

    apply_dark_chart_style(ax)

    return save_chart(
        "flet_age_distribution.png"
    )


def create_a1c_chart(df):
    plt.rcParams["font.family"] = "DejaVu Sans"
    _, ax = plt.subplots(figsize=(8.8, 4.8))

    display_data = df.copy()
    display_data["A1C_display"] = (
        display_data["A1Cresult"]
        .fillna("Не измерен")
        .replace("", "Не измерен")
    )

    order = [
        "Не измерен",
        "Norm",
        ">7",
        ">8",
    ]

    summary = (
        display_data
        .groupby(
            "A1C_display",
            observed=False,
        )["readmitted_30"]
        .mean()
        .reindex(order)
        .fillna(0)
    )

    ax.bar(
        summary.index,
        summary.values * 100,
        color="#22C55E",
    )
    ax.set_title(
        "Доля readmitted <30 по результату HbA1c",
        pad=14,
    )
    ax.set_xlabel("Результат HbA1c")
    ax.set_ylabel("Доля readmitted <30, %")

    apply_dark_chart_style(ax)

    return save_chart(
        "flet_a1c_readmission.png"
    )


def create_risk_group_chart(df):
    plt.rcParams["font.family"] = "DejaVu Sans"
    _, ax = plt.subplots(figsize=(8.8, 4.8))

    order = [
        "низкий риск",
        "средний риск",
        "высокий риск",
    ]

    summary = (
        df
        .groupby(
            "risk_group",
            observed=False,
        )["readmitted_30"]
        .mean()
        .reindex(order)
        .fillna(0)
    )

    ax.bar(
        summary.index,
        summary.values * 100,
        color=[
            "#38BDF8",
            "#FACC15",
            "#FB7185",
        ],
    )
    ax.set_title(
        "Доля readmitted <30 по учебным группам risk_score",
        pad=14,
    )
    ax.set_xlabel("Учебная группа риска")
    ax.set_ylabel("Доля readmitted <30, %")

    apply_dark_chart_style(ax)

    return save_chart(
        "flet_risk_group_rate.png"
    )


def create_top_risk_factors_chart(risk_df):
    plt.rcParams["font.family"] = "DejaVu Sans"

    top_df = risk_df.head(8).copy()

    _, ax = plt.subplots(figsize=(10, 5.8))

    ax.barh(
        top_df["Фактор"],
        top_df["Разница, п.п."],
        color="#A78BFA",
    )
    ax.invert_yaxis()
    ax.set_title(
        "Топ факторов риска readmitted <30",
        pad=14,
    )
    ax.set_xlabel(
        "Изменение риска, процентные пункты"
    )
    ax.set_ylabel("Фактор")

    apply_dark_chart_style(ax)

    return save_chart(
        "flet_top_risk_factors.png"
    )


# ============================================================
# ОСНОВНОЕ ПРИЛОЖЕНИЕ
# ============================================================

def main(page):
    page.title = (
        "Прогнозирование риска госпитализации пациентов с диабетом"
    )
    page.bgcolor = COLORS["page"]
    page.scroll = "auto"
    page.padding = 24

    try:
        page.window_min_width = APP_WIDTH
        page.window_min_height = 760
        page.window_width = 1320
        page.window_height = 850
    except Exception:
        pass

    try:
        page.theme_mode = ft.ThemeMode.DARK
    except Exception:
        page.theme_mode = "dark"

    clean_old_charts()

    state = {
        "df": None,
        "source": None,
        "active_section": None,
    }

    status_text = create_text(
        "Ожидание действия пользователя. "
        "Для начала откройте раздел «Загрузка данных».",
        size=14,
        color=COLORS["green"],
    )

    current_section_text = create_text(
        "Раздел пока не выбран",
        size=17,
        color=COLORS["blue"],
        weight="bold",
    )

    metrics_row = ft.Row(
        controls=[],
        spacing=12,
    )

    content_area = ft.Column(
        controls=[],
        spacing=16,
    )

    charts_area = ft.Column(
        controls=[],
        spacing=16,
    )

    nav_items = {}

    section_names = {
        "load": "Загрузка данных",
        "summary": "Общая сводка",
        "hypotheses": "Проверка гипотез",
        "risk": "Факторы риска",
        "examples": "Примеры высокого риска",
    }

    def set_active_section(section_key):
        state["active_section"] = section_key
        current_section_text.value = (
            section_names.get(
                section_key,
                "Раздел пока не выбран",
            )
        )

        for key, item in nav_items.items():
            is_active = key == section_key

            item["container"].bgcolor = (
                COLORS["active_nav"]
                if is_active
                else COLORS["inactive_nav"]
            )
            item["indicator"].bgcolor = (
                COLORS["green"]
                if is_active
                else COLORS["inactive_nav"]
            )
            item["label"].color = (
                COLORS["white"]
                if is_active
                else COLORS["muted"]
            )
            item["label"].weight = (
                "bold"
                if is_active
                else None
            )

    def make_nav_button(
        section_key,
        label,
        handler,
    ):
        indicator = ft.Container(
            width=5,
            height=24,
            bgcolor=COLORS["inactive_nav"],
            border_radius=4,
        )

        label_control = create_text(
            label,
            size=13,
            color=COLORS["muted"],
        )

        def on_nav_click(event):
            set_active_section(section_key)
            handler(event)

        container = create_container(
            content=ft.Row(
                controls=[
                    indicator,
                    label_control,
                ],
                spacing=8,
            ),
            bgcolor=COLORS["inactive_nav"],
            padding=12,
            border_radius=10,
            on_click=on_nav_click,
        )

        nav_items[section_key] = {
            "container": container,
            "indicator": indicator,
            "label": label_control,
        }

        return container

    def set_status(message, error=False):
        status_text.value = message
        status_text.color = (
            COLORS["red"]
            if error
            else COLORS["green"]
        )
        page.update()

    def clear_content():
        metrics_row.controls.clear()
        content_area.controls.clear()
        charts_area.controls.clear()

    def show_error(message):
        content_area.controls.append(
            make_text_block(
                "Ошибка",
                str(message),
                accent=True,
            )
        )
        set_status(
            "Возникла ошибка. "
            "Проверьте сообщение в текущем разделе.",
            error=True,
        )

    def load_selected_dataset(path=None):
        df, source = load_dataset(path)

        state["df"] = df
        state["source"] = source

        return df, source

    def ensure_data_loaded():
        if state["df"] is None:
            load_selected_dataset()

        return state["df"]

    def render_load_screen(path=None):
        clear_content()

        try:
            df, source = load_selected_dataset(path)

            metrics_row.controls.extend(
                [
                    make_metric_card(
                        "Уникальных пациентов",
                        format_number(len(df)),
                        "пациент-уровневая выборка",
                        "Для каждого пациента оставлен первый доступный эпизод.",
                    ),
                    make_metric_card(
                        "Признаков",
                        df.shape[1],
                        "столбцы аналитической таблицы",
                        "Включены исходные и сформированные признаки.",
                    ),
                    make_metric_card(
                        "Ранняя госпитализация",
                        format_percent(
                            df["readmitted_30"].mean()
                        ),
                        "readmitted = <30",
                        "Доля пациентов с ранней повторной госпитализацией.",
                    ),
                    make_metric_card(
                        "Средний risk_score",
                        f"{df['risk_score'].mean():.2f}",
                        "учебный показатель",
                        "Сумма шести факторов. Не является клинической шкалой.",
                    ),
                ]
            )

            content_area.controls.append(
                make_text_block(
                    "Источник данных",
                    f"Загружен файл: {source.name}\n"
                    f"Полный путь: {source}",
                    accent=True,
                )
            )

            content_area.controls.append(
                create_container(
                    content=ft.Column(
                        controls=[
                            create_text(
                                "Загрузка другого CSV-файла",
                                size=18,
                                color=COLORS["white"],
                                weight="bold",
                            ),
                            create_text(
                                "Скопируйте полный путь к CSV-файлу в Проводнике Windows "
                                "и вставьте его в поле ниже.",
                                size=12,
                                color=COLORS["muted"],
                            ),
                            csv_path_input,
                            ft.Row(
                                controls=[
                                    create_button(
                                        "Загрузить CSV по пути",
                                        load_csv_from_path,
                                    ),
                                    create_button(
                                        "Вернуться к файлу проекта",
                                        load_project_file,
                                    ),
                                ],
                                spacing=12,
                            ),
                        ],
                        spacing=10,
                    ),
                    bgcolor=COLORS["card"],
                    padding=16,
                    border_radius=14,
                    width=APP_WIDTH,
                )
            )

            explanation_rows = [
                {
                    "Показатель": "Уникальных пациентов",
                    "Пояснение": "Количество пациентов после выбора первого эпизода каждого patient_nbr.",
                },
                {
                    "Показатель": "Ранняя госпитализация",
                    "Пояснение": "Доля случаев readmitted = <30 на пациент-уровневой выборке.",
                },
                {
                    "Показатель": "Risk score",
                    "Пояснение": "Учебная сумма шести факторов; HbA1c > 8% в шкалу не включён.",
                },
                {
                    "Показатель": "Выбранный CSV",
                    "Пояснение": "Можно использовать исходный diabetic_data.csv или подготовленный файл проекта.",
                },
            ]

            content_area.controls.append(
                make_table(
                    "Пояснение показателей",
                    columns=[
                        {
                            "title": "Показатель",
                            "key": "Показатель",
                            "width": 280,
                        },
                        {
                            "title": "Краткое пояснение",
                            "key": "Пояснение",
                            "width": 840,
                        },
                    ],
                    rows=explanation_rows,
                    max_rows=10,
                )
            )

            set_status(
                f"Данные успешно подготовлены из файла {source.name}. "
                f"В анализе {format_number(len(df))} уникальных пациентов."
            )

        except Exception as error:
            show_error(error)

        page.update()

    csv_path_input = ft.TextField(
        value="",
        width=790,
    )

    def load_csv_from_path(event):
        """
        Совместимая со старой версией Flet загрузка CSV.
        Пользователь вставляет полный путь к файлу вручную.
        """
        raw_path = str(csv_path_input.value or "").strip()

        if not raw_path:
            show_error(
                "Укажите полный путь к CSV-файлу."
            )
            return

        # Windows часто копирует путь в кавычках.
        selected_path = raw_path.strip('"').strip("'")

        try:
            set_active_section("load")
            render_load_screen(selected_path)
        except Exception as error:
            show_error(error)

    def load_project_file(event):
        set_active_section("load")
        render_load_screen(
            FLET_READY_PATH
        )

    def on_load_click(event):
        render_load_screen(
            state["source"]
            if state["source"] is not None
            else FLET_READY_PATH
        )

    def on_summary_click(event):
        clear_content()

        try:
            df = ensure_data_loaded()

            metrics_row.controls.extend(
                [
                    make_metric_card(
                        "Уникальных пациентов",
                        format_number(len(df)),
                        "пациент-уровневая выборка",
                        "Один первый эпизод на patient_nbr.",
                    ),
                    make_metric_card(
                        "Ранняя госпитализация",
                        format_percent(
                            df["readmitted_30"].mean()
                        ),
                        "целевая переменная",
                        "Доля readmitted = <30.",
                    ),
                    make_metric_card(
                        "Средний возраст",
                        f"{df['age_mid'].mean():.1f}",
                        "лет",
                        "Рассчитан по серединам возрастных интервалов.",
                    ),
                    make_metric_card(
                        "Среднее число лекарств",
                        f"{df['num_medications'].mean():.1f}",
                        "назначений",
                        "Средняя лекарственная нагрузка.",
                    ),
                ]
            )

            summary_rows = [
                {
                    "Показатель": "Количество уникальных пациентов",
                    "Значение": format_number(len(df)),
                    "Пояснение": "Первый эпизод каждого пациента.",
                },
                {
                    "Показатель": "Доля readmitted <30",
                    "Значение": format_percent(
                        df["readmitted_30"].mean()
                    ),
                    "Пояснение": "Целевая переменная анализа.",
                },
                {
                    "Показатель": "Средний возраст",
                    "Значение": f"{df['age_mid'].mean():.2f}",
                    "Пояснение": "Среднее значение age_mid.",
                },
                {
                    "Показатель": "Среднее число диагнозов",
                    "Значение": f"{df['number_diagnoses'].mean():.2f}",
                    "Пояснение": "Диагностическая нагрузка.",
                },
                {
                    "Показатель": "Среднее число лекарств",
                    "Значение": f"{df['num_medications'].mean():.2f}",
                    "Пояснение": "Общая лекарственная нагрузка.",
                },
                {
                    "Показатель": "HbA1c был измерен",
                    "Значение": format_percent(
                        df["a1c_measured"].mean()
                    ),
                    "Пояснение": "Доля пациентов с Norm, >7 или >8.",
                },
            ]

            content_area.controls.append(
                make_table(
                    "Общая сводка по данным",
                    columns=[
                        {
                            "title": "Показатель",
                            "key": "Показатель",
                            "width": 300,
                        },
                        {
                            "title": "Значение",
                            "key": "Значение",
                            "width": 170,
                        },
                        {
                            "title": "Пояснение",
                            "key": "Пояснение",
                            "width": 650,
                        },
                    ],
                    rows=summary_rows,
                    max_rows=20,
                )
            )

            age_chart = create_age_chart(df)
            a1c_chart = create_a1c_chart(df)

            charts_area.controls.extend(
                [
                    make_chart_card(
                        "Возрастная структура",
                        age_chart,
                        "Распределение уникальных пациентов по возрастным интервалам.",
                    ),
                    make_chart_card(
                        "HbA1c и ранняя госпитализация",
                        a1c_chart,
                        "Категория «Не измерен» отделена от реальных результатов анализа.",
                    ),
                ]
            )

            set_status(
                "Общая сводка построена. "
                "Показатели рассчитаны на уровне уникальных пациентов."
            )

        except Exception as error:
            show_error(error)

        page.update()

    def on_hypotheses_click(event):
        clear_content()

        try:
            df = ensure_data_loaded()
            hypotheses_df = calculate_hypotheses(df)

            rows = []

            for _, row in hypotheses_df.iterrows():
                rows.append(
                    {
                        "№": int(row["№"]),
                        "Гипотеза": row["Гипотеза"],
                        "Риск без": format_float(
                            row["Риск без фактора, %"],
                            2,
                        ),
                        "Риск с": format_float(
                            row["Риск с фактором, %"],
                            2,
                        ),
                        "p-value": format_p_value(
                            row["p-value"]
                        ),
                        "V Крамера": format_float(
                            row["Cramer's V"],
                            4,
                        ),
                        "Итог": row["Итог"],
                    }
                )

            content_area.controls.append(
                make_table(
                    "Результаты проверки пяти гипотез",
                    columns=[
                        {
                            "title": "№",
                            "key": "№",
                            "width": 45,
                        },
                        {
                            "title": "Проверяемая гипотеза",
                            "key": "Гипотеза",
                            "width": 410,
                            "max_len": 70,
                        },
                        {
                            "title": "Риск без, %",
                            "key": "Риск без",
                            "width": 120,
                        },
                        {
                            "title": "Риск с, %",
                            "key": "Риск с",
                            "width": 110,
                        },
                        {
                            "title": "p-value",
                            "key": "p-value",
                            "width": 120,
                        },
                        {
                            "title": "V Крамера",
                            "key": "V Крамера",
                            "width": 115,
                        },
                        {
                            "title": "Итог",
                            "key": "Итог",
                            "width": 200,
                        },
                    ],
                    rows=rows,
                    max_rows=10,
                )
            )

            conclusions = [
                "1. HbA1c > 8% не показал статистически значимого повышения риска.",
                "2. Три и более диагноза связаны с повышением риска; связь слабая.",
                "3. Возраст старше 70 лет связан с повышением риска; связь слабая.",
                "4. Значимых различий между мужчинами и женщинами не выявлено.",
                "5. Связь с расовой группой слабая и требует осторожной интерпретации.",
            ]

            content_area.controls.append(
                make_text_block(
                    "Интерпретация",
                    "\n".join(conclusions),
                    accent=True,
                )
            )

            set_status(
                "Проверка гипотез завершена. "
                "Подтверждены гипотезы 2 и 3; "
                "результат гипотезы 5 слабый и интерпретируется осторожно."
            )

        except Exception as error:
            show_error(error)

        page.update()

    def on_risk_click(event):
        clear_content()

        try:
            df = ensure_data_loaded()
            risk_df = calculate_risk_factors(df)

            rows = []

            for _, row in risk_df.iterrows():
                rows.append(
                    {
                        "Фактор": row["Фактор"],
                        "N": format_number(row["N"]),
                        "Риск без": (
                            f"{row['Риск без фактора, %']:.2f}%"
                        ),
                        "Риск с": (
                            f"{row['Риск с фактором, %']:.2f}%"
                        ),
                        "Разница": (
                            f"{row['Разница, п.п.']:.2f} п.п."
                        ),
                        "RR": format_float(
                            row["Относительный риск"],
                            3,
                        ),
                        "OR": format_float(
                            row["Odds Ratio"],
                            3,
                        ),
                    }
                )

            content_area.controls.append(
                make_table(
                    "Ранжирование факторов риска",
                    columns=[
                        {
                            "title": "Фактор",
                            "key": "Фактор",
                            "width": 315,
                        },
                        {
                            "title": "N",
                            "key": "N",
                            "width": 100,
                        },
                        {
                            "title": "Риск без",
                            "key": "Риск без",
                            "width": 125,
                        },
                        {
                            "title": "Риск с",
                            "key": "Риск с",
                            "width": 120,
                        },
                        {
                            "title": "Разница",
                            "key": "Разница",
                            "width": 135,
                        },
                        {
                            "title": "RR",
                            "key": "RR",
                            "width": 95,
                        },
                        {
                            "title": "Odds Ratio",
                            "key": "OR",
                            "width": 125,
                        },
                    ],
                    rows=rows,
                    max_rows=12,
                )
            )

            top_chart = create_top_risk_factors_chart(
                risk_df
            )
            risk_group_chart = create_risk_group_chart(
                df
            )

            charts_area.controls.extend(
                [
                    make_chart_card(
                        "Топ факторов риска",
                        top_chart,
                        "Положительная разница означает повышение доли readmitted <30.",
                    ),
                    make_chart_card(
                        "Учебные группы risk_score",
                        risk_group_chart,
                        "Risk score является учебным эвристическим показателем, а не клинической шкалой.",
                    ),
                ]
            )

            top_3 = risk_df.head(3)

            interpretation_lines = [
                "Наиболее выраженные факторы:"
            ]

            for _, row in top_3.iterrows():
                interpretation_lines.append(
                    f"— {row['Фактор']}: "
                    f"{row['Разница, п.п.']:.2f} п.п."
                )

            content_area.controls.append(
                make_text_block(
                    "Интерпретация факторов",
                    "\n".join(
                        interpretation_lines
                    ),
                    accent=True,
                )
            )

            set_status(
                "Ранжирование факторов выполнено. "
                "HbA1c > 8% не входит в учебный risk_score."
            )

        except Exception as error:
            show_error(error)

        page.update()

    def on_patient_example_click(event):
        clear_content()

        try:
            df = ensure_data_loaded()

            selected_columns = [
                "gender",
                "age",
                "A1Cresult",
                "number_diagnoses",
                "num_medications",
                "number_inpatient",
                "readmitted",
                "risk_score",
                "risk_group",
            ]

            example_df = (
                df
                .sort_values(
                    "risk_score",
                    ascending=False,
                )
                .head(15)
                [selected_columns]
            )

            rows = []

            for _, row in example_df.iterrows():
                rows.append(
                    {
                        "Пол": row["gender"],
                        "Возраст": row["age"],
                        "HbA1c": (
                            row["A1Cresult"]
                            if pd.notna(row["A1Cresult"])
                            else "Не измерен"
                        ),
                        "Диагнозы": row["number_diagnoses"],
                        "Лекарства": row["num_medications"],
                        "Госпитализации": row["number_inpatient"],
                        "Readmitted": row["readmitted"],
                        "Risk score": row["risk_score"],
                        "Группа риска": row["risk_group"],
                    }
                )

            content_area.controls.append(
                make_table(
                    "Примеры уникальных пациентов с высоким risk_score",
                    columns=[
                        {
                            "title": "Пол",
                            "key": "Пол",
                            "width": 85,
                        },
                        {
                            "title": "Возраст",
                            "key": "Возраст",
                            "width": 100,
                        },
                        {
                            "title": "HbA1c",
                            "key": "HbA1c",
                            "width": 100,
                        },
                        {
                            "title": "Диагнозы",
                            "key": "Диагнозы",
                            "width": 100,
                        },
                        {
                            "title": "Лекарства",
                            "key": "Лекарства",
                            "width": 110,
                        },
                        {
                            "title": "Прошлые госпит.",
                            "key": "Госпитализации",
                            "width": 145,
                        },
                        {
                            "title": "Readmitted",
                            "key": "Readmitted",
                            "width": 115,
                        },
                        {
                            "title": "Risk score",
                            "key": "Risk score",
                            "width": 110,
                        },
                        {
                            "title": "Группа риска",
                            "key": "Группа риска",
                            "width": 190,
                        },
                    ],
                    rows=rows,
                    max_rows=15,
                )
            )

            content_area.controls.append(
                make_text_block(
                    "Пояснение",
                    "Risk score суммирует шесть признаков: "
                    "три и более диагноза, возраст старше 70 лет, "
                    "предыдущие стационарные обращения, экстренные обращения, "
                    "высокую лекарственную нагрузку и использование инсулина. "
                    "Показатель создан для учебной демонстрации и не является "
                    "клинически валидированной шкалой.",
                    accent=True,
                )
            )

            set_status(
                "Показаны примеры уникальных пациентов "
                "с наибольшим учебным risk_score."
            )

        except Exception as error:
            show_error(error)

        page.update()

    header = create_container(
        content=ft.Column(
            controls=[
                create_text(
                    "Прогнозирование риска госпитализации пациентов с диабетом",
                    size=28,
                    color=COLORS["white"],
                    weight="bold",
                ),
                create_text(
                    "Итоговая аттестационная работа • "
                    "Николаев Даниил Сергеевич • КББО-03-23",
                    size=15,
                    color=COLORS["muted"],
                ),
            ],
            spacing=8,
        ),
        bgcolor=COLORS["header"],
        padding=22,
        border_radius=18,
        width=APP_WIDTH,
    )

    buttons = create_container(
        content=ft.Row(
            controls=[
                make_nav_button(
                    "load",
                    "Загрузка данных",
                    on_load_click,
                ),
                make_nav_button(
                    "summary",
                    "Общая сводка",
                    on_summary_click,
                ),
                make_nav_button(
                    "hypotheses",
                    "Проверка гипотез",
                    on_hypotheses_click,
                ),
                make_nav_button(
                    "risk",
                    "Факторы риска",
                    on_risk_click,
                ),
                make_nav_button(
                    "examples",
                    "Примеры высокого риска",
                    on_patient_example_click,
                ),
            ],
            spacing=10,
        ),
        bgcolor=COLORS["card"],
        padding=14,
        border_radius=14,
        width=APP_WIDTH,
    )

    current_section_block = create_container(
        content=ft.Row(
            controls=[
                ft.Container(
                    width=6,
                    height=38,
                    bgcolor=COLORS["blue"],
                    border_radius=4,
                ),
                ft.Column(
                    controls=[
                        create_text(
                            "Текущий раздел",
                            size=12,
                            color=COLORS["muted"],
                        ),
                        current_section_text,
                    ],
                    spacing=2,
                ),
            ],
            spacing=12,
        ),
        bgcolor=COLORS["card_2"],
        padding=14,
        border_radius=12,
        width=APP_WIDTH,
    )

    main_column = ft.Column(
        controls=[
            header,
            spacer(10),
            make_instruction_block(),
            spacer(10),
            buttons,
            spacer(8),
            current_section_block,
            spacer(10),
            metrics_row,
            content_area,
            charts_area,
            spacer(10),
            make_status_block(status_text),
        ],
        spacing=8,
    )

    page.add(
        create_container(
            content=main_column,
            bgcolor=COLORS["page"],
            padding=0,
            border_radius=0,
            width=APP_WIDTH,
        )
    )

    page.update()


if __name__ == "__main__":
    # Приложение запускается в оконном режиме.
    # Для совместимости со старой версией Flet загрузка другого CSV
    # выполняется через ввод полного пути к файлу.
    try:
        ft.app(
            target=main,
            assets_dir="assets",
        )
    except Exception:
        ft.app(
            target=main,
            assets_dir="assets",
            view=ft.WEB_BROWSER,
        )
