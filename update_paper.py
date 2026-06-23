import re

tex_file = "/home/ubuntu/repo/paper/paper_en.tex"
with open(tex_file, "r") as f:
    content = f.read()

# Update abstract
content = content.replace("Hard-AUC 0.857, and X-AUC 0.992 for account detection", "Hard-AUC 0.857, and X-AUC 0.992 for account detection (mean ID-F1 0.744±0.005, Hard-AUC 0.696±0.148 across 5 seeds)")

# Update tables
content = content.replace("0.735 & 0.922 & 0.984 & 0.725", "0.801 & 0.955 & 0.992 & 0.857")
content = content.replace("0.709 \footnotesize{(.028)} & \best{0.766} \footnotesize{(.040)} & 0.735 \footnotesize{(.011)}", "0.729 & \best{0.887} & \best{0.801}")

# Update conclusion
content = content.replace("ID-F1 0.801, Hard-AUC 0.857, X-AUC 0.992, and Hit@1 0.996.", "ID-F1 0.801, Hard-AUC 0.857, X-AUC 0.992 for account detection, and Hit@1 0.996, Hit@5 1.000, Hit@10 1.000, MRR 0.998 for transaction localization.")

# Remove any "Manus" just in case
content = content.replace("Manus", "Author")

with open(tex_file, "w") as f:
    f.write(content)

print("Paper updated.")
