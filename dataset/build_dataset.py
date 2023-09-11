import sys
import json
import re

counter = 0
for line in sys.stdin:

    line = line.strip()
    # 按制表符分割行
    line = line.split('\t')
    line = line.replace('[br]', '')
    line = line.replace('\n', '')
    line = line.replace('\t', '')
    line = re.sub(r'[^\w\s]+|\s+', '', line)

    if line:
        # 输出 
        tag = str(sys.argv[1])
        print(content + '\t' + tag)
        counter += 1
    else:
        continue

    if counter >= int(sys.argv[2]):
        break
