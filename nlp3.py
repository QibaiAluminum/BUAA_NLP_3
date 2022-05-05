import jieba, os, re
import numpy as np
from gensim import corpora, models

def D_dispose():
    fileName = "./dic_train_data.txt"
    if not os.path.exists('./dic_train_data.txt'):
        outputs = open(fileName, 'w', encoding='UTF-8')
        DSRoot = "./data"
        catalog = "inf.txt"
        with open(os.path.join(DSRoot, catalog), "r", encoding='utf-8') as f:
            all_files = f.readline().split(",")
            print(all_files)
        for name in all_files:
            with open(os.path.join(DSRoot, name + ".txt"), "r", encoding='utf-8') as f:
                file_read = f.readlines()
                train_num = len(file_read)
                choice_index = np.random.choice(len(file_read), train_num, replace=False)
                sw = ["的", "了", "在", "是", "我", "有", "和", "就",
			"不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
			"会", "着", "没有", "看", "好", "自己", "这","罢" ,"这",'在','又','在','得','那','他','她','不','而','道','与','之','见','却','问','可','但'
                      ,'没','啦','给','来','既','叫','只','中','么','便'
                      ,'听','为','跟','个','甚','下','还','过','向','如此'
                      ,'已','位','对','如何','将','岂','哪','似','以免','均'
                      ,'虽然','即','由','再','使','从','麽','其实','阿','被']
                for train in choice_index[0:train_num]:
                    line = file_read[train]
                    line = re.sub('\s', '', line)
                    line = re.sub('[\u0000-\u4DFF]', '', line)
                    line = re.sub('[\u9FA6-\uFFFF]', '', line)
                    line= re.sub('[\u9FA6-\uFFFF]', '', line)
                    for a in sw:
                        line= re.sub(a, '', line)
                    if len(line) == 0:
                        continue
                    seg_list = list(jieba.cut(line, cut_all=False))
                    line_seg = ""
                    for term in seg_list:
                        line_seg += term + " "
                    outputs.write(line_seg.strip() + '\n')
        outputs.close()
        print("处理完原始数据")


if __name__ == "__main__":
    D_dispose()
    #整理成gensim需要的输入格式
    fr = open('./dic_train_data.txt', 'r', encoding='utf-8')
    train = []
    for line in fr.readlines():
        line = [word.strip() for word in line.split(' ')]
        train.append(line)
    #训练LDA模型
    dictionary = corpora.Dictionary(train)
    # corpus是把每本小说ID化后的结果，每个元素是新闻中的每个词语，在字典中的ID和频率
    corpus = [dictionary.doc2bow(text) for text in train]
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=16)
    topic_list_lda = lda.print_topics(16)
    print("16本小说主题分布为：\n")
    for topic in topic_list_lda:
        print(topic)

    fileTest = "./dic_test_data.txt"
    news_test = open(fileTest, 'r', encoding='UTF-8')
    test = []
    for line in news_test:
        line = [word.strip() for word in line.split(' ')]
        test.append(line)
    for text in test:
        corpus_test = dictionary.doc2bow((text))
    corpus_test = [dictionary.doc2bow(text) for text in test]
    topics_test = lda.get_document_topics(corpus_test)

    for i in range(10):
        print("测试用例"+str(i)+'的主题分布为：'+str(topics_test[i]))

    fr.close()
    news_test.close()
