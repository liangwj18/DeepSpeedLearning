# https://cloud.tencent.com/developer/article/1510591

import time
import psutil
from matplotlib import pyplot as plt

def getinfo():
    mem = psutil.virtual_memory()
    mem_total = mem.total
    mem_free = mem.free
    mem_used = mem.used
    mem_percent = mem.percent
    cpu = psutil.cpu_percent(1)
    return mem_total, mem_free, mem_used, mem_percent, cpu

info_total = {
    "mem_total": [],
    "mem_free": [],
    "mem_used": [],
    "mem_percent": [],
    "cpu_percent": [],
}


if __name__ == '__main__':
    while True:
        try:
            print(getinfo())
            t = int(time.time())
            time.sleep(1)
        except Exception as e:
            print(e)


    # plt.plot(all_cpu)
    # plt.show()
