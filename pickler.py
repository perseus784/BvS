import pickle as p
def write(x,file_name):
    file=open(file_name,"wb")
    p.dump(x,file)
    file.close()
    print("Data serialization complete..........")
    pass
def retrive(filename):
    print("retrieving data..........")
    file=open(filename,"rb")
    val=p.load(file)
    file.close()
    print("Data deserialization complete..........")
    return val

if __name__=="__main__":
    write("hello world")
    print(retrive())
