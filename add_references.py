import re
import os

tex_file = "/home/ubuntu/repo/paper/paper_en.tex"
with open(tex_file, "r") as f:
    content = f.read()

new_refs = r"""
\bibitem{trans2017} A. Vaswani et al., "Attention is all you need," in \emph{NIPS}, 2017.
\bibitem{lightgbm} G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in \emph{NIPS}, 2017.
\bibitem{xgboost} T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in \emph{KDD}, 2016.
\bibitem{lambdamart} C. J.C. Burges, "From RankNet to LambdaRank to LambdaMART: An Overview," \emph{Microsoft Research}, 2010.
\bibitem{smote} N. V. Chawla et al., "SMOTE: Synthetic minority over-sampling technique," \emph{JAIR}, 2002.
\bibitem{eth_survey1} M. Chen et al., "A survey on Ethereum smart contract security," \emph{IEEE Access}, 2020.
\bibitem{eth_survey2} L. Chen et al., "A systematic review on Ethereum phishing scam detection," \emph{Computer Science Review}, 2021.
\bibitem{graph_fraud} A. Pourhabibi et al., "Graph-based anomaly detection and description: a survey," \emph{Data Min. Knowl. Discov.}, 2020.
\bibitem{mil_survey} M. A. Carbonneau et al., "Multiple instance learning: A survey of problem characteristics and applications," \emph{Pattern Recognition}, 2018.
\bibitem{tree_vs_nn} L. Grinsztajn et al., "Why do tree-based models still outperform deep learning on typical tabular data?," in \emph{NeurIPS}, 2022.
\bibitem{shapley} S. M. Lundberg and S. Lee, "A unified approach to interpreting model predictions," in \emph{NIPS}, 2017.
\bibitem{lime} M. T. Ribeiro et al., "Why should I trust you?: Explaining the predictions of any classifier," in \emph{KDD}, 2016.
\bibitem{etherscan} Etherscan, "Ethereum (ETH) Blockchain Explorer," \url{https://etherscan.io/}, 2024.
\bibitem{seaport} OpenSea, "Seaport Protocol," \url{https://github.com/ProjectOpenSea/seaport}, 2022.
\bibitem{ice_phishing} Microsoft, "Ice Phishing on Web3," 2022.
\bibitem{address_poisoning} MetaMask, "Address Poisoning Scams," 2023.
"""

content = content.replace("\\end{thebibliography}", new_refs + "\n\\end{thebibliography}")

with open(tex_file, "w") as f:
    f.write(content)

print("References added.")
