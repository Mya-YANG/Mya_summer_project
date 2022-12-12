
import os


root_path = ''

def contrastDir(file_dir):

    png_list = []
    txt_list = []
    for root, dirs, files in os.walk(file_dir+'mega_json'):

        for file in files:
            if os.path.splitext(file)[1] == '.json':
                png_list.append(os.path.splitext(file)[0])
    for root, dirs, files in os.walk(file_dir+'true_label'):

        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                txt_list.append(os.path.splitext(file)[0])


#    diff = set(txt_list).difference(set(png_list))
#    print(len(diff))
#    for name in diff:
#        print(name + ".txt"+'has No corresponding image file')

    diff2 = set(png_list).difference(set(txt_list))
    print(len(diff2))
    for name in diff2:
#        print(name + ".png"+" has No corresponding TXT file")

        os.remove(file_dir+'mega_json/'+name+'.json')
    return png_list,txt_list

if __name__ == '__main__':

    contrastDir(root_path)
