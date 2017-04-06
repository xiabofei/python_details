#encoding=utf8
import jieba

seg_list = jieba.cut('我来到知春路东华合创大厦', cut_all=True)
print "\t".join(seg_list)

seg_list = jieba.cut('我来到知春路东华合创大厦', cut_all=False)
print "\t".join(seg_list)
