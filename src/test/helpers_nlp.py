from vncorenlp import VnCoreNLP

# lần đầu: tải model về thư mục mặc định
nlp = VnCoreNLP(annotators=["wseg"], max_heap_size='-Xmx500m')

text = "Tôi thích ăn tôm nên muốn gọi món khác bạn có món nào không"
sents = nlp.sent_tokenize(text, no_punc=True)
print(sents)

# đừng quên đóng
nlp.close()