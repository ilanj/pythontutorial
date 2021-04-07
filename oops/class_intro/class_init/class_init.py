class Testing():
    def __init__(self):
        pass

    def get_no(self, no):
        self.no = no
        self.sq()

    def sq(self):
        self.sq = self.no * self.no
        self.sq_sq()

    def sq_sq(self):
        self.sq2 = self.sq * self.sq


# run = Testing("tested")
# run.printing()



