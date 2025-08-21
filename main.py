import re
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
