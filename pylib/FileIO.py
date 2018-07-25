# Xi Peng, Jun 2016
import os, sys, shutil, random
import numpy as np

def ListSubfolderInFolder(path):
    return [f for f in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path,f))]

def ListFileInFolder(path,format):
    list = []
    for root, dirs, files in os.walk(path):
        for file in sorted(files):
            if file.endswith(format):
                list.append(root+file)
    return list


def ListFileInFolderRecursive(path,format):
    list = []
    for root, dirs, files in os.walk(path):
        for fold in dirs:
            files = os.listdir(root+fold)
            for file in sorted(files):
                if file.endswith(format):
                    list.append(fold+'/'+file)
    return list

def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def WriteLineToFile(path,lines):
    with open(path, 'w') as fd:
        for line in lines:
            fd.write(line + '\n')

def WriteLineToFileShuffle(path,lines):
    random.shuffle(lines)
    with open(path, 'w') as fd:
        for line in lines:
            fd.write(line + '\n')

def ReadFloatFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            line = line.rstrip('\n').split(' ')
            line2 = [float(line[i]) for i in range(len(line))]
            lines.append(line2)
    lines = np.array(lines)
    return lines

def WriteFloatToFile(path,lines):
    with open(path,'w') as fd:
        print lines.shape
        print lines.ndim 

def DeleteThenCreateFolder(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


if __name__=='__main__':
    print 'Python IO Lib by Xi Peng'
