timeline = [("0" + str(int(i / 60)) if i / 60 < 10 else str(int(i / 60))) + ":" + (
"0" + str(i % 60) if i % 60 < 10 else str(i % 60)) for i in range(30, 1470, 30)]
timeline =  timeline # one for weekday and weekendday
print(timeline)

timeline2 = [(str(i / 60) if i / 60 > 9 else "0" + str(i / 60)) + ":" + (str(i % 60) if i % 60 > 9 else "0" + str(i % 60)) for i in range(0, 1440, 30)]
print(timeline2)

timerange = range(0, 1440, 30)
print(timerange)

timerange2 = ['%s:%s' % (h, m) for h in ([00] + list(range(1, 24))) for m in ('00', '30')]
print(timerange2)
#"{0:0=2d}".format(a)

# for i in range(30, 1440, 30):
#     #print(i/60)
#     if i/60 < 10:
#         print("0" + str(int(i / 60)))
#         #print(str(int(i/60)))
#     else:
#         print(str(int(i / 60)))

