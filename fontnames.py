f = open('fontnames.txt', 'r+')
read_data = f.readlines()
out=[]
#print (read_data)
f.close()
f = open('fnames.txt', 'w')
for a in read_data:
#  f.write(a[-4:-1])
    if (a[-4:-1]=="ttf"):
        s = a.split(" ")
        out.append(s[-1])
        f.write(s[-1])
f.close()