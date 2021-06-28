from xml.dom.minidom import parse
import xml.dom.minidom
import pandas as pd
# git config --global http.sslVerify false

DOMTree = xml.dom.minidom.parse("data/SMP2019_ECISA_Dev.xml")
collection = DOMTree.documentElement

docs = collection.getElementsByTagName("Doc")

result = {
    "text": [],
    "pre_text": [],
    "last_pre": [],
    "label": []
}


for doc in docs:

    sentences = doc.getElementsByTagName("Sentence")
    pre = ""
    last = ""
    flag = True
    text = None
    label = None
    for sentence in sentences:
        if sentence.hasAttribute("label"):
            text = sentence.childNodes[0].data
            label = sentence.getAttribute("label")
            flag = False
        if flag:
            pre += sentence.childNodes[0].data + "###"
        if not flag:
            last += sentence.childNodes[0].data + "###"

    result["text"].append(text)
    result["pre_text"].append(pre)
    result["last_pre"].append(last)
    result["label"].append(label)


df = pd.DataFrame(result)
df.to_csv("data/dev.csv")


