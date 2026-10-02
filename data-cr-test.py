"""
สร้างข้อมูลสังเคราะห์ (synthetic data) 7 ตาราง ตาม schema ในเอกสารโจทย์ Student Learning Analytics

** ข้อมูลทั้งหมดสุ่มขึ้นมาเองจากชื่อคอลัมน์และคะแนนเต็มในเอกสารเท่านั้น **
** ไม่ได้อิงค่าจริง/สถิติจริงของข้อมูลอาจารย์เลย ใช้สำหรับทดสอบโค้ดเท่านั้น **

วิธีใช้:
    pip install numpy pandas
    python generate_synthetic_data.py

ผลลัพธ์: โฟลเดอร์ synthetic_data/ มีไฟล์ CSV 7 ไฟล์ ชื่อเหมือนในโจทย์
"""

import os

import numpy as np
import pandas as pd

# ---------------------------- ตั้งค่า ----------------------------
SEED = 42              # เปลี่ยนเลขนี้ได้ถ้าอยากได้ข้อมูลชุดใหม่
N_STUDENTS = 120       # จำนวนนักศึกษาทั้งหมด
OUT_DIR = "synthetic_data"
ADD_MESSY = True       # True = ใส่ข้อมูลสกปรกเล็กน้อย (ค่าว่าง/ซ้ำ) ไว้ทดสอบขั้น cleaning
N_WEEKS = 15

rng = np.random.default_rng(SEED)


# ---------------------------- ฟังก์ชันช่วย ----------------------------
def clip_round(x, low, high, decimals=2):
    return np.round(np.clip(x, low, high), decimals)


def make_student_ids(n):
    # รูปแบบรหัสสมมติ 10 หลัก (ไม่ใช่รหัสจริง)
    return [f"6{rng.integers(0, 10)}{i:08d}"[:10] for i in range(1, n + 1)]


def make_attendance(ids, ability, base_rate):
    """ability สูง -> มีโอกาสเข้าเรียนสูงขึ้น"""
    p = np.clip(base_rate + 0.12 * ability, 0.35, 0.99)
    data = rng.random((len(ids), N_WEEKS)) < p[:, None]
    df = pd.DataFrame(data.astype(int), columns=[f"Week{i}" for i in range(1, N_WEEKS + 1)])
    df.insert(0, "StudentID", ids)
    return df


def make_score_table(ids, ability, attend_ratio, components, weights):
    """
    components: dict ชื่อคอลัมน์ -> คะแนนเต็ม
    weights:    dict ชื่อคอลัมน์ -> น้ำหนักในคะแนนสรุป (รวมกัน = 1)
    """
    n = len(ids)
    df = pd.DataFrame({"StudentID": ids})
    ratios = {}
    for col, full in components.items():
        # สัดส่วนคะแนน = ความสามารถ + การเข้าเรียน + noise
        r = 0.68 + 0.10 * ability + 0.10 * (attend_ratio - 0.8) + rng.normal(0, 0.09, n)
        r = np.clip(r, 0.05, 1.0)
        ratios[col] = r
        df[col] = clip_round(r * full, 0, full)

    final = sum(weights[c] * (df[c] / components[c]) for c in components) * 100
    df["ScoreGrade(100Points)"] = clip_round(final + rng.normal(0, 0.8, n), 0, 100)

    # คอลัมน์ Quiz เป็น int ตามโจทย์
    for col in components:
        if col.startswith("Quiz"):
            df[col] = df[col].round().astype(int)
    return df


def add_mess(df, id_col="StudentID", missing_frac=0.01, dup_n=2):
    """ใส่ค่าว่างแบบสุ่ม และแถวซ้ำเล็กน้อย"""
    df = df.copy()
    value_cols = [c for c in df.columns if c != id_col]
    mask = rng.random((len(df), len(value_cols))) < missing_frac
    for j, c in enumerate(value_cols):
        df[c] = df[c].astype("float64").where(~mask[:, j], np.nan)
    dups = df.sample(dup_n, random_state=int(rng.integers(0, 10_000)))
    return pd.concat([df, dups], ignore_index=True)


# ---------------------------- สร้างข้อมูล ----------------------------
ids = make_student_ids(N_STUDENTS)
ability = rng.normal(0, 1, N_STUDENTS)  # ตัวแปรแฝง "ความสามารถ" ที่ผู้เรียนไม่เห็น

# 1) ID_Group.csv
gpa = clip_round(2.7 + 0.45 * ability + rng.normal(0, 0.25, N_STUDENTS), 1.0, 4.0)
id_group = pd.DataFrame(
    {
        "StudentID": ids,
        "Group": rng.integers(1, 4, N_STUDENTS),  # กลุ่ม 1-3
        "GPA": gpa,
    }
)

# 2) SJ111777 (Attendance + Score)
att_111 = make_attendance(ids, ability, base_rate=0.82)
ratio_111 = att_111.iloc[:, 1:].mean(axis=1).to_numpy()
score_111 = make_score_table(
    ids,
    ability,
    ratio_111,
    components={
        "Tasks(100Points)": 100,
        "Quiz(150Points)": 150,
        "Project(100Points)": 100,
        "Midterm(150Points)": 150,
        "Final(150Points)": 150,
    },
    weights={
        "Tasks(100Points)": 0.15,
        "Quiz(150Points)": 0.20,
        "Project(100Points)": 0.15,
        "Midterm(150Points)": 0.25,
        "Final(150Points)": 0.25,
    },
)

# 3) SJ220777 (Attendance + Score)
att_220 = make_attendance(ids, ability, base_rate=0.78)
ratio_220 = att_220.iloc[:, 1:].mean(axis=1).to_numpy()
score_220 = make_score_table(
    ids,
    ability,
    ratio_220,
    components={
        "Tasks(100Points)": 100,
        "Quiz(90Points)": 90,
        "Project(90Points)": 90,
        "Midterm(180Points)": 180,
        "Final(180Points)": 180,
    },
    weights={
        "Tasks(100Points)": 0.15,
        "Quiz(90Points)": 0.10,
        "Project(90Points)": 0.10,
        "Midterm(180Points)": 0.325,
        "Final(180Points)": 0.325,
    },
)

# 4) SJ333777 (Attendance) และ S333777 (Score)
att_333 = make_attendance(ids, ability, base_rate=0.85)
ratio_333 = att_333.iloc[:, 1:].mean(axis=1).to_numpy()
score_333 = make_score_table(
    ids,
    ability,
    ratio_333,
    components={
        "Tasks(100Points)": 100,
        "Project(100Points)": 100,
        "Midterm(100Points)": 100,
        "Final(100Points)": 100,
    },
    weights={
        "Tasks(100Points)": 0.20,
        "Project(100Points)": 0.20,
        "Midterm(100Points)": 0.30,
        "Final(100Points)": 0.30,
    },
)

# ---------------------------- ใส่ข้อมูลสกปรก (ถ้าเปิด) ----------------------------
tables = {
    "ID_Group.csv": id_group,
    "SJ111777_Attendance15weeks.csv": att_111,
    "SJ111777_Score.csv": score_111,
    "SJ220777_Attendance15weeks.csv": att_220,
    "SJ220777_Score.csv": score_220,
    "S333777_Score.csv": score_333,
    "SJ333777_Attendance15weeks.csv": att_333,
}

if ADD_MESSY:
    for name in ("SJ111777_Score.csv", "SJ220777_Score.csv", "S333777_Score.csv"):
        tables[name] = add_mess(tables[name])
    # ผู้เรียนบางคนไม่มีข้อมูลในบางวิชา (เช่น ถอน/ไม่ได้ลง) เพื่อทดสอบการ merge
    drop_ids = rng.choice(ids, size=6, replace=False)
    tables["SJ220777_Score.csv"] = tables["SJ220777_Score.csv"][
        ~tables["SJ220777_Score.csv"]["StudentID"].isin(drop_ids[:3])
    ]
    tables["SJ333777_Attendance15weeks.csv"] = tables["SJ333777_Attendance15weeks.csv"][
        ~tables["SJ333777_Attendance15weeks.csv"]["StudentID"].isin(drop_ids[3:])
    ]

# ---------------------------- บันทึกไฟล์ ----------------------------
os.makedirs(OUT_DIR, exist_ok=True)
for name, df in tables.items():
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"บันทึก {path}  ({len(df)} แถว, {df.shape[1]} คอลัมน์)")

print("\nเสร็จแล้ว (ข้อมูลทั้งหมดเป็นข้อมูลสังเคราะห์)")