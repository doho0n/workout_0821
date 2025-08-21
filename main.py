import re
import io
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# PDF parsing
try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except Exception:
    PYPDF2_AVAILABLE = False

############################
# Helpers
############################

INBODY_KR_KEYS = {
    "weight": ["체중", "몸무게", "Weight"],
    "smm": ["골격근량", "SMM", "Skeletal Muscle Mass"],
    "bfm": ["체지방량", "Body Fat Mass"],
    "pbf": ["체지방률", "PBF", "Percent Body Fat"],
    "bmr": ["기초대사량", "BMR", "Basal Metabolic Rate"],
    "height": ["신장", "키", "Height"],
}

ACTIVITY_FACTORS = {
    "좌식/운동 거의 안함": 1.2,
    "가벼운 활동(주 1-3회)": 1.375,
    "보통 활동(주 3-5회)": 1.55,
    "높은 활동(주 6-7회)": 1.725,
    "매우 높음(육체노동/2회훈련)": 1.9,
}

GOAL_MAP = {
    "다이어트(감량)": -500,
    "유지": 0,
    "벌크업(증량)": 500,
}

SEX_PROTEIN_RULE = {
    "남성": (1.5, 2.0),
    "여성": (1.2, 1.5),
}

FOODS = {
    "탄수화물": ["현미밥", "오트밀", "고구마", "감자", "통밀빵", "쌀국수"],
    "단백질": ["닭가슴살", "계란", "두부", "연어", "대구", "그릭요거트", "프로틴쉐이크"],
    "지방": ["아몬드", "아보카도", "올리브오일", "호두", "땅콩버터"],
}

############################
# Parsing functions
############################

def extract_numbers_near_keywords(text: str, keywords: list) -> Optional[float]:
    # Find the first occurrence of any keyword and grab a number near it (same line or next tokens)
    for kw in keywords:
        # Pattern: keyword + optional chars + number (int/float)
        pattern = rf"{re.escape(kw)}\D*(\d+(?:[\.,]\d+)?)"
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            val = m.group(1).replace(",", ".")
            try:
                return float(val)
            except Exception:
                pass
    return None


def parse_inbody_pdf(file_bytes: bytes) -> Dict[str, Optional[float]]:
    if not PYPDF2_AVAILABLE:
        return {k: None for k in INBODY_KR_KEYS}
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        txt = "\n".join(page.extract_text() or "" for page in reader.pages)
        res = {k: extract_numbers_near_keywords(txt, v) for k, v in INBODY_KR_KEYS.items()}
        return res
    except Exception:
        return {k: None for k in INBODY_KR_KEYS}

############################
# Calculation functions
############################

def mifflin_st_jeor(sex: str, weight_kg: float, height_cm: float, age: int) -> float:
    # weight kg, height cm, age years
    if sex == "남성":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    return bmr


def tdee_from_activity(bmr: float, activity_label: str) -> float:
    return bmr * ACTIVITY_FACTORS.get(activity_label, 1.55)


def plan_macros(tdee: float, goal_label: str, sex: str, weight_kg: float, fat_ratio: float = 0.20,
                protein_per_kg: Optional[float] = None) -> Dict[str, float]:
    goal_delta = GOAL_MAP.get(goal_label, 0)
    target_kcal = max(1000.0, tdee + goal_delta)  # guardrail

    # protein grams: choose lower bound if not provided
    low, high = SEX_PROTEIN_RULE[sex]
    ppk = protein_per_kg if protein_per_kg else low
    protein_g = ppk * weight_kg
    protein_kcal = protein_g * 4

    fat_kcal = target_kcal * fat_ratio
    fat_g = fat_kcal / 9

    carb_kcal = max(0.0, target_kcal - protein_kcal - fat_kcal)
    carb_g = carb_kcal / 4

    return {
        "target_kcal": round(target_kcal, 0),
        "protein_g": round(protein_g, 0),
        "fat_g": round(fat_g, 0),
        "carb_g": round(carb_g, 0),
        "fat_ratio": fat_ratio,
        "protein_per_kg": ppk,
    }


def weekly_weight_change_hint(goal_label: str) -> str:
    if goal_label == "다이어트(감량)":
        return "일주일에 약 -0.5kg 변화를 목표로 하세요. 1주 뒤 변화를 보고 칼로리를 -100~-150kcal 추가 조정할 수 있어요."
    elif goal_label == "벌크업(증량)":
        return "일주일에 약 +0.5kg 변화를 목표로 하세요. 1주 뒤 변화를 보고 칼로리를 +100~+150kcal 추가 조정할 수 있어요."
    return "체중 유지가 목표예요. 주 단위로 컨디션과 퍼포먼스를 점검하세요."

############################
# Meal generator
############################

def suggest_meals(macros: Dict[str, float]) -> Tuple[pd.DataFrame, str]:
    # Very simple templating to meet macros approximately
    target = macros
    meals = []

    # Template 3 meals + 1 snack
    # We’ll split macros roughly: 30% breakfast, 35% lunch, 25% dinner, 10% snack
    splits = [("아침", 0.30), ("점심", 0.35), ("저녁", 0.25), ("간식", 0.10)]

    for name, r in splits:
        p = round(target["protein_g"] * r)
        c = round(target["carb_g"] * r)
        f = round(target["fat_g"] * r)
        foods = [
            np.random.choice(FOODS["단백질"]),
            np.random.choice(FOODS["탄수화물"]),
            np.random.choice(FOODS["지방"]),
        ]
        meals.append({"식사": name, "권장 단백질(g)": p, "권장 탄수화물(g)": c, "권장 지방(g)": f, "예시 음식": ", ".join(foods)})

    df = pd.DataFrame(meals)
    note = (
        "단백질은 살코기/계란/유청 등으로 우선 충족하고, 나머지 칼로리는 자유롭게 구성해도 됩니다.\n"
        "(외식/간식 포함). 단, 목표 칼로리를 초과하지 않도록 ‘영양성분표’ 기준으로 기록하세요."
    )
    return df, note

############################
# Training plan generator (simple)
############################

def training_plan(goal_label: str, days: int = 5) -> pd.DataFrame:
    # Minimal, universal routine suggestions
    templates = {
        "A": [
            ("하체+코어", "스쿼트 변형 4x6-10, 루마니안 데드리프트 3x8-10, 런지 3x10/측, 플랭크 3x45초"),
            ("가슴+삼두", "벤치프레스 4x6-10, 인클라인 덤벨프레스 3x8-12, 케이블 플라이 3x12-15, 로프 푸시다운 3x12-15"),
            ("등+이두", "랫풀다운 4x8-12, 바벨 로우 3x6-10, 시티드 로우 3x10-12, 컬 3x12-15"),
            ("어깨", "오버헤드프레스 4x6-10, 레터럴 레이즈 4x12-20, 리어델트 3x12-20"),
            ("전신(볼륨 낮게)", "데드리프트 3x5, 풀업/보조 3x최대, 딥스/보조 3x최대, 행잉 레그레이즈 3x10-15"),
        ],
        "B": [
            ("전신(초보)", "레그프레스 3x10-15, 머신 체스트프레스 3x10-12, 랫풀다운 3x10-12, 레터럴 레이즈 3x15-20, 케이블 컬 2x12-15, 트라이셉스 익스텐션 2x12-15"),
            ("코어+유산소", "하이퍼익스텐션 3x12-15, 케이블 크런치 3x12-15, 사이드 플랭크 3x30초, 인터벌 15-20분"),
        ],
    }

    plan = templates["A"] if days >= 4 else templates["B"]
    if goal_label == "다이어트(감량)":
        cardio = "저강도 유산소 20-30분(웨이트 후) 주 3-5회"
    elif goal_label == "벌크업(증량)":
        cardio = "가벼운 컨디셔닝 10-15분, 회복 위주"
    else:
        cardio = "선호에 따라 주 2-3회 20분"

    df = pd.DataFrame(plan, columns=["요일/세션", "루틴(예시)"])
    df.loc[len(df)] = ["유산소", cardio]
    return df

############################
# UI
############################

st.set_page_config(page_title="Fitness Planner – Diet/Bulk", page_icon="💪", layout="wide")
st.title("식단 & 운동 추천 – 인바디 기반 (Diet/Bulk)")

left, right = st.columns([1, 1])

with left:
    st.header("1) 인바디 업로드 또는 직접 입력")
    up = st.file_uploader("인바디 PDF/CSV (선택)", type=["pdf", "csv"])

    sex = st.selectbox("성별", ["남성", "여성"])
    age = st.number_input("나이", min_value=14, max_value=90, value=28)

    height_cm = st.number_input("키(cm)", min_value=120, max_value=220, value=175)
    weight_kg = st.number_input("체중(kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1)

    smm = None
    pbf = None

    if up is not None:
        if up.type == "text/csv":
            try:
                df = pd.read_csv(up)
                # Try common columns
                for col in df.columns:
                    lc = col.lower()
                    if "weight" in lc or "체중" in col:
                        weight_kg = float(df[col].iloc[0])
                    if "height" in lc or "신장" in col:
                        height_cm = float(df[col].iloc[0])
                    if "pbf" in lc or "체지방률" in col:
                        pbf = float(df[col].iloc[0])
                st.success("CSV에서 값 일부를 불러왔어요.")
            except Exception as e:
                st.warning(f"CSV 파싱 실패: {e}")
        elif up.type == "application/pdf":
            b = up.read()
            parsed = parse_inbody_pdf(b)
            if parsed.get("height"):
                height_cm = float(parsed["height"]) if parsed["height"] > 3 else float(parsed["height"]) * 100
            if parsed.get("weight"):
                weight_kg = float(parsed["weight"])
            if parsed.get("pbf"):
                pbf = float(parsed["pbf"])
            if parsed.get("smm"):
                smm = float(parsed["smm"])
            st.info("PDF에서 추정 값을 불러왔어요. 값이 맞는지 확인해 주세요.")

    activity = st.selectbox("활동 수준", list(ACTIVITY_FACTORS.keys()), index=2)
    goal = st.radio("목표", list(GOAL_MAP.keys()), index=0, horizontal=True)

    # Protein setting
    low, high = SEX_PROTEIN_RULE[sex]
    ppk = st.slider("단백질(체중 x g)", min_value=low, max_value=high, value=low, step=0.1)
    fat_ratio = st.slider("지방 비율(총칼로리 대비)", min_value=0.15, max_value=0.30, value=0.20, step=0.01)

with right:
    st.header("2) 칼로리/매크로 계산")
    bmr = mifflin_st_jeor(sex, weight_kg, height_cm, age)
    tdee = tdee_from_activity(bmr, activity)
    macros = plan_macros(tdee, goal, sex, weight_kg, fat_ratio=fat_ratio, protein_per_kg=ppk)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("목표 칼로리", f"{int(macros['target_kcal'])} kcal")
    m2.metric("단백질", f"{int(macros['protein_g'])} g")
    m3.metric("탄수화물", f"{int(macros['carb_g'])} g")
    m4.metric("지방", f"{int(macros['fat_g'])} g")

    st.caption("Tip: 일주일 간 체중 변화를 관찰하고, 목표와 다르면 100~150kcal 단위로 미세 조정하세요.")

    df_meal, note = suggest_meals(macros)
    st.subheader("예시 식단(가이드)")
    st.dataframe(df_meal, use_container_width=True)
    st.write(note)

    st.subheader("운동 루틴(예시)")
    days = st.slider("주당 웨이트 횟수", 2, 6, 5)
    df_train = training_plan(goal, days)
    st.table(df_train)

    # Downloads
    st.subheader("다운로드")
    plan = {
        "입력": {
            "성별": sex,
            "나이": age,
            "키_cm": height_cm,
            "체중_kg": weight_kg,
            "활동수준": activity,
            "목표": goal,
            "단백질_배수": ppk,
            "지방_비율": fat_ratio,
            "인바디_PBF_추정": pbf,
            "인바디_SMM_추정": smm,
        },
        "계산": macros,
        "예시식단": df_meal.to_dict(orient="records"),
        "예시루틴": df_train.to_dict(orient="records"),
    }
    json_bytes = io.BytesIO()
    json_bytes.write(pd.Series(plan).to_json(force_ascii=False, indent=2).encode("utf-8"))
    json_bytes.seek(0)
    st.download_button("개인 플랜(JSON) 다운로드", data=json_bytes, file_name="fitness_plan.json", mime="application/json")

st.divider()
st.markdown(
    """
**가이드 요약**
- 목표 칼로리 = TDEE ± 500kcal (감량은 -500, 증량은 +500)
- 지방은 총칼로리의 약 20% 수준으로 설정
- 단백질은 남성 1.5~2.0 g/kg, 여성 1.2~1.5 g/kg 중에서 현재 수행능력에 맞게 선택
- 탄수화물은 남은 칼로리로 채우기
- 주 1회 체중 변화를 확인해 ±0.5kg 정도를 목표로 소폭 조정
"""
)
