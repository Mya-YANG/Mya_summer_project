import get_img2 as gi

colors=((220,20,60),(230,230,250),(65,105,225),(0,255,127),(0,191,255),(124,252,0),(255,255,0),(255,140,0))

thr=(0.01,0.02,0.03,0.04,0.1,0.2,0.6,0.8)

for i in range(len(thr)):
    
    if (i == 0):
        
        file_dir1 = 'ori_pic'
        
        file_dir2 = 'multi_mege_' + str(i)
        
        gi.multi_mege_vis('RPXD0004',thr[i],file_dir1,file_dir2,colors[i])
        
    else:
        
        file_dir1 = file_dir2
        
        file_dir2 = 'multi_mege_' + str(i)
        
        gi.multi_mege_vis('RPXD0004',thr[i],file_dir1,file_dir2,colors[i])
        
        gi.multi_mega_video('RPXD0004',file_dir2)




