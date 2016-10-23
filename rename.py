#coding: utf-8
import os
import string
import shutil

root=os.path.join(os.getcwd(),"classes")
newroot=os.path.join(os.getcwd(),"classes_rename")
if not os.path.exists(newroot):
    os.makedirs(newroot)
# 假设当前目录有许多文件夹，每个文件夹中是同一个人的一些人脸
seq=0
for folder in os.listdir(root):
    seq+=1
    ind=0
    for img in os.listdir(os.path.join(root,folder)):
        if img.split('.')[-2].split('_')[-1] !="anno":
            srcpath=os.path.join(root,folder,img)
            ind+=1
            newname=img.split('.')[-1]
            newname=str(seq)+"-"+str(ind)+".jpg"
            newpath=os.path.join(newroot,newname)
            shutil.copy(srcpath,newpath)
       