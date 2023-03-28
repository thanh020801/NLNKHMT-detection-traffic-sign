import os
path = 'full_size_data'

classes = 11
for i in range(1,classes):
    pathq = os.path.join(path,str(i))
    images = os.listdir(pathq)
    # print(pathq)
    # if len(images) > 0:
    j = 0
    # if len(images) > 0:
        # print(images[1])
        # print(os.path.join(pathq, images[1]))
    for filename in images:
        
        # if filename.endswith(".jpg"):
        #     os.rename(os.path.join(pathq, filename), os.path.join(pathq, str(i)+"_000"+str(j)+".jpg"))
        #     j+=1
        try:
            
            check_jpg = False
            check_txt = False
            # if filename.endswith(".jpg"):
            #     # print(os.path.join(pathq, filename))
            #     check_jpg = True
            #     os.rename(os.path.join(pathq, filename), os.path.join(pathq, str(i)+"_000"+str(j)+".jpg"))
            #     j+=1
            if filename.endswith(".txt"):
                print(os.path.join(pathq, filename))
                check_txt = True
                os.rename(os.path.join(pathq, filename), os.path.join(pathq, str(i)+"_000"+str(j)+".txt"))
                j+=1
            # if check_jpg and check_txt:
            #     j+=1
        except:
            # print(os.path.join(pathq, filename))
            print('ko the doc file')
    