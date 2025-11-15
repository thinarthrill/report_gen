import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from hdbscan import HDBSCAN
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# --------------------------
# 1. Настройки
# --------------------------
NEG_WORDS = set("""
ужас плохо плохо работает ужасно недоволен злится возмущен ошибка не работает 
не грузит не открывается не получается не могу медленно зависает сбой 
проблема жалоба возмутительно невозможно бесит раздражает долго слишком медленно 
""".split())

MIN_TEXT_LEN = 10


# --------------------------
# 2. Функции очистки текста
# --------------------------
def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"[^а-яёa-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --------------------------
# 3. Вычисление sentiment
# (очень простая модель)
# --------------------------
def sentiment_score(text: str):
    words = text.split()
    neg = sum(1 for w in words if w in NEG_WORDS)
    score = neg / (len(words) + 1)
    return score  # от 0 до ~0.4


# --------------------------
# 4. Кластеризация тем
# --------------------------
def cluster_topics(df):
    vect = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X = vect.fit_transform(df["clean_text"])

    clusterer = HDBSCAN(min_cluster_size=10, metric='euclidean')
    labels = clusterer.fit_predict(X)

    df["topic"] = labels
    return df, vect


# --------------------------
# 5. Pareto анализ по темам
# --------------------------
def pareto_topics(df):
    topic_counts = df["topic"].value_counts().reset_index()
    topic_counts.columns = ["topic", "count"]
    topic_counts["cum%"] = topic_counts["count"].cumsum() / topic_counts["count"].sum() * 100
    return topic_counts


# --------------------------
# 6. “Красная зона клиентов”
# --------------------------
def hot_clients(df, threshold=0.15):
    tmp = df.groupby("client_id")["sentiment"].mean().reset_index()
    return tmp[tmp["sentiment"] >= threshold].sort_values("sentiment", ascending=False)


# --------------------------
# 7. Генерация HTML отчёта
# --------------------------
def build_report(topic_stats, hot, df, filename="report.html"):
    fig1 = px.bar(topic_stats, x="topic", y="count", title="Количество обращений по темам")
    fig2 = px.line(topic_stats, x="topic", y="cum%", title="Pareto (накопительный %)")
    fig3 = px.scatter(hot, x="client_id", y="sentiment", title="Клиенты красной зоны")

    html = f"""
    <html>
    <head><meta charset="utf-8"><title>AI Report</title></head>
    <body>
    <h1>Аналитический отчёт по обращениям</h1>

    <h2>1. Pareto тем недовольства</h2>
    {fig1.to_html(full_html=False)}
    {fig2.to_html(full_html=False)}

    <h2>2. Клиенты “красной зоны”</h2>
    {fig3.to_html(full_html=False)}

    <h2>3. TOP фразы</h2>
    {df['clean_text'].head(30).to_list()}
    </body>
    </html>
    """

    Path(filename).write_text(html, encoding="utf-8")
    print(f"✔ Отчёт сохранён: {filename}")


# --------------------------
# 8. Основной запуск
# --------------------------
def run_agent(input_file: str):
    print(f"Загружаю данные: {input_file}")
    df = pd.read_excel(input_file) if input_file.endswith(".xlsx") else pd.read_csv(input_file)

    # Определяем вероятные колонки
    text_col = [c for c in df.columns if "тема" in c.lower() or "опис" in c.lower() or "коммент" in c.lower()]
    client_col = [c for c in df.columns if "клиент" in c.lower() or "абонент" in c.lower() or "id" in c.lower()]

    if not text_col:
        raise ValueError("Не найдена колонка с текстом.")
    if not client_col:
        df["client_id"] = np.arange(len(df))
    else:
        df["client_id"] = df[client_col[0]]

    df["clean_text"] = df[text_col[0]].astype(str).apply(clean_text)
    df = df[df["clean_text"].str.len() > MIN_TEXT_LEN].reset_index(drop=True)

    # Sentiment
    df["sentiment"] = df["clean_text"].apply(sentiment_score)

    # Clustering
    df, vect = cluster_topics(df)

    # Pareto
    topic_stats = pareto_topics(df)

    # Hot clients
    hot = hot_clients(df)

    # HTML report
    build_report(topic_stats, hot, df, "AI_ContactCenter_Report.html")


if __name__ == "__main__":
    run_agent("input.xlsx")   # путь к твоему файлу
