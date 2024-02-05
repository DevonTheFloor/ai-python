import plotly.express as px

def draw_message_scores(pham, pspam, height=200, width=400):
    fig = px.bar(
        [{"Label": "ham", "Prob": pham}, {"Label": "spam", "Prob": pspam}],
        y="Label",
        x="Prob",
        color="Label",
        height=height,
        width=width,
        color_discrete_map={"ham": "seagreen", "spam": "tomato"},
    )
    fig.update_layout(showlegend=False, margin=dict(r=0, l=0, t=0, b=0))
    fig.update_xaxes(range=(0, 1))
    return fig

def draw_word_scores(words, line_height=50, width=400):
    if len(words) > 0:
        words = sorted(words.items(), key=lambda x: x[1][1])
    else:
        words = [("      ", (0.5, 0.5))]
    words_data = [
        {"Word": word, "Label": "ham", "Prob": pham}
        for word, (pham, _) in words
    ] + [
        {"Word": word, "Label": "spam", "Prob": pspam}
        for word, (_, pspam) in words
    ]
    fig = px.bar(
        words_data,
        y="Word",
        x="Prob",
        color="Label",
        height=line_height * len(words) if len(words) > 1 else 150,
        width=width,
        color_discrete_map={"ham": "seagreen", "spam": "tomato"},
    )
    fig.update_layout(margin=dict(r=0, l=0, t=0, b=0))
    fig.update_xaxes(range=(0, 1))
    return fig