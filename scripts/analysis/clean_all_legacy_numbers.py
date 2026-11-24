# -*- coding: utf-8 -*-
"""
clean_all_legacy_numbers.py
"""

def clean_chapter_4():
    path = "Scripts/chapter_4_results_and_discussion.md"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Replacements of metrics
    text = text.replace("0.5894", "0.4814")
    text = text.replace("0.5819", "0.5252")
    text = text.replace("0.2374", "0.4772")
    text = text.replace("0.3178", "0.4634")
    text = text.replace("0.9231", "0.8731")
    text = text.replace("0.9184", "0.9053")
    text = text.replace("0.7412", "0.8806")
    text = text.replace("0.8105", "0.8756")
    text = text.replace("0.2365", "0.4763")
    text = text.replace("0.2413", "0.4811")
    text = text.replace("0.7398", "0.8830")
    text = text.replace("0.7425", "0.8807")
    text = text.replace("27.51%", "24.51%")
    text = text.replace("26.50%", "24.97%")
    text = text.replace("0.12%", "23.52%")
    text = text.replace("17.50%", "19.47%")
    text = text.replace("49.73%", "39.83%")
    text = text.replace("49.29%", "43.90%")
    text = text.replace("22.98%", "39.96%")
    text = text.replace("27.46%", "39.53%")
    text = text.replace("-0.3445", "-0.0480")
    text = text.replace("-0.3520", "-0.0042")
    text = text.replace("+0.0804", "-0.0138")
    text = text.replace("-0.2716", "-0.0181")
    text = text.replace("+0.0765", "-0.0178")
    text = text.replace("455,902.35", "345,402.31")
    text = text.replace("465,061.63", "365,728.99")
    text = text.replace("571,942.37", "634,205.97")
    text = text.replace("2,123,185.89", "629,411.78")
    text = text.replace("9,159.28", "20,326.68")
    text = text.replace("$455.9k", "$345.4k")
    text = text.replace("$465.1k", "$365.7k")

    # Clean RQ1-3 summary in 4.6
    old_rq1 = "While handcrafted relational features yield slight negative raw AP deltas on novel entities, they generate the lowest economic loss"
    new_rq1 = "Handcrafted relational features in Stage 2 deliver a statistically significant uplift to $\\text{AP} = 0.5252$ ($\\Delta_{2-1} = +0.0438$, $p = 0.00175$) and generate the lowest economic loss (\\$345.4k, saving \\$20,326.68)"
    text = text.replace(old_rq1, new_rq1)

    old_rq2 = "Replacing tabular feature engineering with isolated continuous message passing leads to catastrophic failure ($\\Delta_{3-2} = -0.3445$ AP)"
    new_rq2 = "Replacing tabular feature engineering with continuous message passing leads to performance degradation ($\\Delta_{3-2} = -0.0480$ AP, $\\text{AP} = 0.4772$)"
    text = text.replace(old_rq2, new_rq2)

    old_rq3 = "Pre-registered hypothesis $H_3$ is **confirmed** ($\\Delta_{4-3} = +0.0804$ AP, $\\text{LCB}_{0.95} = +0.0712$)."
    new_rq3 = "Pre-registered hypothesis $H_3$ is **not supported** ($\\Delta_{4-3} = -0.0138$ AP, $p_{\\text{adj}} = 0.00175$). Joint alternating functional updates failed to improve upon the frozen Res-GNN and fell below the tabular baseline ($\\Delta_{4-1} = -0.0181$ AP, $H_4$ not supported)."
    text = text.replace(old_rq3, new_rq3)

    # Clean duplicate ASCII table blocks if any remain
    # Remove second occurrence of table 4.1 if repeated
    lines = text.splitlines()
    cleaned_lines = []
    skip_duplicate = False
    table_count = 0
    for line in lines:
        if "| Architectural Stage / Model Variant" in line:
            table_count += 1
            if table_count > 1:
                skip_duplicate = True
                continue
        if skip_duplicate:
            if line.startswith("+") and ("––" in line or "--" in line):
                skip_duplicate = False
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Cleaned Chapter 4.")

def clean_chapter_5():
    path = "Scripts/chapter_5_conclusion_and_recommendations.md"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    text = text.replace("0.5894", "0.4814")
    text = text.replace("0.5819", "0.5252")
    text = text.replace("0.2374", "0.4772")
    text = text.replace("0.3178", "0.4634")
    text = text.replace("455,902.35", "345,402.31")
    text = text.replace("465,061.63", "365,728.99")
    text = text.replace("2,123,185.89", "629,411.78")
    text = text.replace("27.51%", "24.51%")
    text = text.replace("26.50%", "24.97%")
    text = text.replace("17.50%", "19.47%")

    # Remove second duplicate occurrence of table 5.2 if repeated
    lines = text.splitlines()
    cleaned_lines = []
    skip_duplicate = False
    table_count = 0
    for line in lines:
        if "| Architectural Metric" in line or ("| Metric / Dimension" in line and table_count > 0):
            skip_duplicate = True
            continue
        if skip_duplicate:
            if line.startswith("+") and ("––" in line or "--" in line):
                skip_duplicate = False
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Cleaned Chapter 5.")

if __name__ == "__main__":
    clean_chapter_4()
    clean_chapter_5()
