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
#f_pred = open(r'../log/text_save/val_predict_sort.txt',"r")
f_pred = open(r'../../ICDAR_TASK2_new8/log/text_save/val_predict_sort.txt',"r")
f_pred1 = open(r'../../ICDAR_TASK2_new6/log/text_save/val_predict_sort1.txt',"r")
f_pred2 = open(r'../../ICDAR_TASK2_new9/log/text_save/val_predict_sort.txt',"r")
f_pred3 = open(r'../../ICDAR_TASK2_new10/log/text_save/val_predict_sort.txt',"r")
f_pred4 = open(r'../../ICDAR_TASK2_new11/log/text_save/val_predict_sort.txt',"r")
f_real = open(pre_data_dir + '/test_data/val_words_gt_sort.txt',"r")
charset='! "#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_`abcdefghijklmnopqrstuvwxyz~¡¢£©°É•€★'
pred_line = f_pred.readline()
pred_line1 = f_pred1.readline()
pred_line2 = f_pred2.readline()
pred_line3 = f_pred3.readline()
pred_line4 = f_pred4.readline()
real_line = f_real.readline()
count_same_case_insensitive = 0
long_same = 0
Flag_same = 0
char_statis = {} 
for e in charset:
    char_statis[e] = character_statis(e)   
num_classes=len(charset)
count_all_sample = 0
pred_line = f_pred.readline()
real_line = f_real.readline()  
pred_line1 = f_pred1.readline()
pred_line2 = f_pred2.readline()
pred_line3 = f_pred3.readline()
pred_line4 = f_pred4.readline()

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



while pred_line:
    count_all_sample += 1
    pred_line = str(pred_line)
    predict_string = pred_line.split(',')[1]
    predict_string = re.sub('[\r\n\t]', '', predict_string)
    
    pred_line1 = str(pred_line1)
    predict1_string = pred_line1.split(',')[1]
    predict1_string = re.sub('[\r\n\t]', '', predict1_string)
    
    pred_line2 = str(pred_line2)
    predict2_string = pred_line2.split(',')[1]
    predict2_string = re.sub('[\r\n\t]', '', predict2_string)
    
    pred_line3 = str(pred_line3)
    predict3_string = pred_line3.split(',')[1]
    predict3_string = re.sub('[\r\n\t]', '', predict3_string)
    
    
    pred_line4 = str(pred_line4)
    predict4_string = pred_line4.split(',')[1]
    predict4_string = re.sub('[\r\n\t]', '', predict4_string)
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

     
    predict_string = predict_string.upper()
    predict1_string = predict1_string.upper()
    predict2_string = predict2_string.upper()
    predict3_string = predict3_string.upper()
    predict4_string = predict4_string.upper()
    real_string = real_string.upper()
    if predict_string == real_string:
        Flag_same = 0
    
    if predict1_string == real_string:
        Flag_same = 1

    if predict2_string == real_string:
        Flag_same = 1

    if predict3_string == real_string:
        Flag_same = 1

    #if predict4_string == real_string:
     #   Flag_same = 1
    count_same_case_insensitive += Flag_same
    
    Flag_same = 0
    pred_line = f_pred.readline()
    pred_line1 = f_pred1.readline()
    pred_line2 = f_pred2.readline()
    pred_line3 = f_pred3.readline()
    pred_line4 = f_pred4.readline()
    real_line = f_real.readline()  
update_value_by_new_statis()
f_pred.close()
f_pred1.close()
f_pred2.close()
f_pred3.close()
f_pred4.close()
f_real.close()
print("count_all_sample",count_all_sample)
print("count_same_case_insensitive", count_same_case_insensitive)
print("acc_case_insensitive", 1.0*count_same_case_insensitive/count_all_sample)
