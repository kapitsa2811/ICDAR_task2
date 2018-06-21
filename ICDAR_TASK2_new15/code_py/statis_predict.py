import re
class character_statis:
    def __init__(self, character):
        self.character = character
        self.total_num = 0
        self.total_acc_num = 0
        self.total_acc_rate = 0.0
        self.map_to_every_character_num = {}
        for e in charset:
            self.map_to_every_character_num[e] = 0
 
           
pre_data_dir = '/home/sjhbxs/Data/data_coco_task2/ICDAR_TASK2_new2'
f_pred = open(r'../log/text_save/val_predict_sort.txt',"r")
f_real = open(pre_data_dir + '/test_data/val_words_gt_sort.txt',"r")
charset='! "#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_`abcdefghijklmnopqrstuvwxyz~¡¢£©°É•€★'
pred_line = f_pred.readline()
real_line = f_real.readline()
count_same_case_insensitive = 0
long_same = 0
char_statis = {} 
for e in charset:
    char_statis[e] = character_statis(e)   
num_classes=len(charset)
count_all_sample = 0
pred_line = f_pred.readline()
real_line = f_real.readline()  

def count_num_for_character_statis(pre_string,real_string):
    if len(pre_string) == len(real_string):
        for e_pre, e_real in zip(pre_string, real_string):
            char_statis[e_real].map_to_every_character_num[e_pre] += 1


def update_value_by_new_statis():
    for e in charset:
        char_statis[e].total_num = 0
        char_statis[e].total_acc_num = char_statis[e].map_to_every_character_num[e]
        for c in charset:
            char_statis[e].total_num += char_statis[e].map_to_every_character_num[c]
        if char_statis[e].total_num != 0:
            char_statis[e].total_acc_rate = 1.0 * char_statis[e].total_acc_num / char_statis[e].total_num

num_pre_big = 0
num_pre_small = 0

while pred_line:
    count_all_sample += 1
    pred_line = str(pred_line)
    predict_string = pred_line.split(',')[1]
    predict_string = re.sub('[\r\n\t]', '', predict_string)
    
    real_line = str(real_line)
    if ',' in real_line:
        real_string = real_line.split(',')[1]
        real_string = re.sub('[\r\n\t]', '', real_string)
    else:
        print("I do not have ,", real_line)
        #real_line = f_real.readline()
        continue
    if len(real_string) == 0:
        print("I do not have char:", real_line.split(','[0]))
    if len(real_string) != 0 and real_string[0] == '|':
        real_string = real_line.split('|')[1]
        real_string = re.sub('[\r\n\t]', '', real_string)

    if len(predict_string) == len(real_string):
        long_same += 1
        count_num_for_character_statis(predict_string, real_string)
    
    if len(predict_string) != len(real_string):
        if len(predict_string) > len(real_string):
            num_pre_big += 1
        else:
            num_pre_small += 1
        print(predict_string,real_string)
    predict_string = predict_string.upper()
    real_string = real_string.upper()
    if predict_string == real_string:
        count_same_case_insensitive += 1

    pred_line = f_pred.readline()
    real_line = f_real.readline()  

update_value_by_new_statis()
f_pred.close()
f_real.close()
print("count_all_sample",count_all_sample)
print("count_same_case_insensitive", count_same_case_insensitive)
print("acc_case_insensitive", 1.0*count_same_case_insensitive/count_all_sample)
print("long_same", long_same)
print("long_same_rate", 1.0*long_same/count_all_sample)
print("num pre is longer:", num_pre_big)
print("num_pre is shorter:", num_pre_small)
for e in charset:
    print(e, " ", char_statis[e].total_acc_rate, char_statis[e].total_num)
