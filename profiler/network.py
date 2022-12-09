import time
import sys

# proc/net/dev中每一项的含义是：
# bytes: The total number of bytes of data transmitted or received by the interface.（接口发送或接收的数据的总字节数）
# packets: The total number of packets of data transmitted or received by the interface.（接口发送或接收的数据包总数）
# errs: The total number of transmit or receive errors detected by the device driver.（由设备驱动程序检测到的发送或接收错误的总数）
# drop: The total number of packets dropped by the device driver.（设备驱动程序丢弃的数据包总数）
# fifo: The number of FIFO buffer errors.（FIFO缓冲区错误的数量）
# frame: The number of packet framing errors.（分组帧错误的数量）
# colls: The number of collisions detected on the interface.（接口上检测到的冲突数）
# compressed: The number of compressed packets transmitted or received by the device driver. (This appears to be unused in the 2.2.15 kernel.)（设备驱动程序发送或接收的压缩数据包数）
# carrier: The number of carrier losses detected by the device driver.（由设备驱动程序检测到的载波损耗的数量）
# multicast: The number of multicast frames transmitted or received by the device driver.（设备驱动程序发送或接收的多播帧数）
#
# 示例如下：$ cat proc/net/dev
# Inter-|   Receive                                                |  Transmit
#  face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
#     lo: 31443571708 40488459    0    0    0     0          0         0 31443571708 40488459    0    0    0     0       0          0

if len(sys.argv) > 1:
    INTERFACE = sys.argv[1]
else:
    INTERFACE = 'ens6f0'
STATS = []
print('Interface:', INTERFACE)


def rx():
    ifstat = open('/proc/net/dev').readlines()
    for interface in ifstat:
        if INTERFACE in interface:
            stat = float(interface.split()[1])
            STATS[0:] = [stat]
            return stat


def tx():
    ifstat = open('/proc/net/dev').readlines()
    for interface in ifstat:
        if INTERFACE in interface:
            stat = float(interface.split()[9])
            STATS[1:] = [stat]
            return stat


print(f"""
'In        Out'
{rx()}
{tx()}
""")

while True:
    print(STATS)
    time.sleep(1)
    STATS_ORIGIN = list(STATS)
    rx()
    tx()
    RX = float(STATS[0])
    RX_O = STATS_ORIGIN[0]
    TX = float(STATS[1])
    TX_O = STATS_ORIGIN[1]
    RX_RATE = round((RX - RX_O) / 1024 / 1024, 3)
    TX_RATE = round((TX - TX_O) / 1024 / 1024, 3)
    print(f"{RX_RATE} MB     {TX_RATE} MB")

# 模拟 scp
# 0.006 MB     0.015 MB
# [180333128141.0, 126441262120.0]
# 0.163 MB     15.011 MB
# [180333299389.0, 126457002300.0]
# 0.704 MB     69.159 MB
# [180334037666.0, 126529520474.0]
# 0.691 MB     67.14 MB
# [180334762672.0, 126599921594.0]
# 0.666 MB     65.897 MB
# [180335460714.0, 126669019276.0]
# 0.664 MB     64.076 MB
# [180336156545.0, 126736207822.0]
# 0.726 MB     65.457 MB
# [180336918092.0, 126804844729.0]
# 0.911 MB     84.206 MB
# [180337873852.0, 126893141163.0]
# 0.738 MB     70.036 MB
# [180338647489.0, 126966579675.0]
# 0.789 MB     73.924 MB
# [180339474736.0, 127044094367.0]
# 0.786 MB     70.792 MB
# [180340299355.0, 127118325647.0]
# 0.634 MB     59.586 MB
# [180340963823.0, 127180806253.0]
# 0.252 MB     22.027 MB
# [180341228116.0, 127203903537.0]
# 0.005 MB     0.012 MB
# [180341233858.0, 127203916459.0]