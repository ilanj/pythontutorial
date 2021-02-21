class Test:
    def data_for_a_document(self):
        path = "dataset_5classes/entertainment/"
        filename = "008.txt"
        f = open(path + filename)
        r = f.read()
        test_doc = []
        test_doc.append(r)
        return test_doc

if __name__ == '__main__':
    obj=Test()
    test =obj.data_for_a_document()
    print(type(test))
