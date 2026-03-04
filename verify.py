import os
import random
import numpy as np
from eval_core.cal_score_fast import ImageSimilarity

random.seed(42)

# 是否相似度分数不重要，正确生成图片更重要
def verify_what_matters(img_dir="/root/dev/baseline/data/output_eval"):
    imgs = [img_dir+'/'+img for img in os.listdir(img_dir)]
    scorer = ImageSimilarity()
    scores = [0] * len(imgs)
    for i in range(0, len(imgs)):
        for j in range(i+1, len(imgs)):
            s = scorer.comprehensive_similarity(imgs[i], imgs[j])["comprehensive"]
            scores[i] += s
            scores[j] += s
    print(f"{len(imgs)} number")
    scores = np.array(scores)
    print(scores)
    scores = scores / len(imgs)
    print(scores)
    print(f"mean: {np.mean(scores)}, std: {np.std(scores)}")
    return scores

import torch
from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('/root/dev/baseline/swift_output/datav0_5_norm_aug/v3-20260222-121751/checkpoint-4500-merged', trust_remote_code=True)
# s = "\\frac{\\partial^2 u}{\\partial t^2} -\\nabla\\cdot(c\\nabla u) + au = f\\\\\\begin{pmatrix}\\int_{-\\infty}^{\\infty} e^{-x^2} dx&\\sum_{k=1}^{10}\\sin(kx)\\\\\\lim_{x\\to 0}\\frac{\\sin(x)}{x}&\\prod_{n=1}^{5} n\\\\\\sum_{i=1}^{3}\\int_{0}^{i} x^{2} dx&\\sum_{j=1}^{4}\\frac{1}{j}\\\\\\end{pmatrix} =\\begin{pmatrix}\\sqrt{\\pi}&2.34\\\\1&120\\\\\\frac{7}{3}&2.08\\\\\\end{pmatrix}"
# output = torch.LongTensor([59, 37018, 35702, 37420, 61, 17, 575, 15170, 59, 37420, 259, 61, 17, 92, 481, 59, 77, 370, 4260, 59, 50853, 1337, 1699, 370, 4260, 575, 8, 488, 7906, 284, 282, 81351, 7265, 90, 5187, 2555, 11035, 396, 15159, 30529, 258, 36958, 92, 61, 35702, 258, 36958, 92, 384, 87210, 87, 61, 17, 92, 13822, 5, 59, 1242, 15159, 74, 28, 16, 92, 47822, 16, 15, 11035, 15940, 5969, 87, 8, 81351, 4659, 15159, 87, 59, 983, 220, 15, 11035, 37018, 35702, 15940, 2075, 9139, 90, 87, 24778, 59, 19748, 15159, 77, 28, 16, 92, 47822, 20, 92, 308, 81351, 1242, 15159, 72, 28, 16, 92, 47822, 18, 11035, 396, 15159, 15, 92, 47822, 72, 92, 856, 47822, 17, 92, 13822, 5, 59, 1242, 15159, 73, 28, 16, 92, 47822, 19, 11035, 37018, 90, 16, 15170, 73, 92, 81351, 408, 90, 5187, 2555, 92, 284, 59, 7265, 90, 5187, 2555, 11035, 26888, 35702, 2493, 24778, 17, 13, 18, 19, 3422, 16, 5, 16, 17, 15, 81351, 37018, 90, 22, 15170, 18, 24778, 17, 13, 15, 23, 81351, 408, 90, 5187, 2555, 92]).unsqueeze(0)

# response = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
# response = '\\frac{\\partial^2 u}{\\partial t^2} -\\nabla\\cdot(c\\nabla u) + au = f\\\\\\begin{pmatrix}\\int_{-\\infty}^{\\infty} e^{-x^2} dx&\\sum_{k=1}^{10}\\sin(kx)\\\\\\lim_{x\\to 0}\\frac{\\sin(x)}{x}&\\prod_{n=1}^{5} n\\\\\\sum_{i=1}^{3}\\int_{0}^{i} x^{2} dx&\\sum_{j=1}^{4}\\frac{1}{j}\\\\\\end{pmatrix} =\\begin{pmatrix}\\sqrt{\\pi}&2.34\\\\1&120\\\\\\frac{7}{3}&2.08\\\\\\end{pmatrix}'
# 注入
# sample00087.png
outputs = torch.LongTensor([59, 37018, 35702, 37420, 61, 17, 575, 15170, 59, 37420,\
    259, 61, 17, 92, 481, 59, 77, 370, 4260, 59, 50853, 1337, 1699, 370, 4260, 575, \
    8, 488, 7906, 284, 282, 81351, 7265, 90, 5187, 2555, 11035, 396, 15159, 30529, \
    258, 36958, 92, 61, 35702, 258, 36958, 92, 384, 87210, 87, 61, 17, 92, 13822, 5, \
    59, 1242, 15159, 74, 28, 16, 92, 47822, 16, 15, 11035, 15940, 5969, 87, 8, 81351, \
    4659, 15159, 87, 59, 983, 220, 15, 11035, 37018, 35702, 15940, 2075, 9139, 90, 87, \
    24778, 59, 19748, 15159, 77, 28, 16, 92, 47822, 20, 92, 308, 81351, 1242, 15159, 72, \
    28, 16, 92, 47822, 18, 11035, 396, 15159, 15, 92, 47822, 72, 92, 856, 47822, 17, 92, \
    13822, 5, 59, 1242, 15159, 73, 28, 16, 92, 47822, 19, 11035, 37018, 90, 16, 15170, 73, \
    92, 81351, 408, 90, 5187, 2555, 92, 284, 59, 7265, 90, 5187, 2555, 11035, 26888, 35702, \
    2493, 24778, 17, 13, 18, 19, 3422, 16, 5, 16, 17, 15, 81351, 37018, 90, 22, 15170, 18, \
    24778, 17, 13, 15, 23, 81351, 408, 90, 5187, 2555, 92]).unsqueeze(0)


img_dir = "/root/dev/data/datav3/gimgs"
imgs = [img_dir+'/'+img for img in os.listdir(img_dir)]
for img in random.choices(imgs, k=100):
    os.system(f"cp {img} /root/dev/baseline/eval/testset/v3_100")
