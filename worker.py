import os

def time():
    p = os.popen('uptime')
    x=p.read()
    return x


os.system("mv ./temp/* ./backup/")
for i in range(1):
    os.system("python init_net.py")
    os.system("python train_net.py 5000 0.001")
    p=os.popen("python run_in_test.py")
    x=p.read()
    cmd="cp ./net_param ./temp/"+x.replace(", grad_fn=<MseLossBackward0>)","").replace("tensor(","")
    print("cmd: "+cmd)
    os.system(cmd)
