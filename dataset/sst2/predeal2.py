def parse_standard_file(fin_path,fout_path):
    error_cnt = 0
    with open(fout_path,'w') as fout:
        with open(fin_path) as fin:
            for line in fin.readlines():
                line2 = line.replace("\t\t","\t")
                fout.write(line2)
                
parse_standard_file("./train.txt","./train2.txt")
