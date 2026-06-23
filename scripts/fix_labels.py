import re

tex_file = "/home/ubuntu/repo/paper/paper_en.tex"
with open(tex_file, "r") as f:
    content = f.read()

# Fix tab:comp
content = content.replace(r"\label{tab:loc}", r"\label{tab:comp}", 1) # This is risky, let's be precise
# Let's find the table with "Account-level comparison" and add the label
content = re.sub(r"(caption\{Account-level comparison vs. published SOTA and MIL pooling baselines. ID = in-domain; X = zero-shot cross-domain PTXPhish; X\$_\{hard\}\$ = AUC vs. hard PTXPhish benign senders.\})", r"\1\n\\label{tab:comp}", content)

# Fix tab:prf
content = re.sub(r"(caption\{In-domain account-level Precision/Recall/F1 \(threshold 0.5, mean over 5 seeds\). UnifiedTMIL achieves the highest recall and F1.\})", r"\1\n\\label{tab:prf}", content)

with open(tex_file, "w") as f:
    f.write(content)

print("Labels fixed.")
