import functools


class Tester(object):
    
    def __init__(self):
        super(Tester, self).__init__()

    def sf_cv(self, statsfunction, Xgroups, Ygroups, **kwargs):
        results = []
        for X, Y in zip(Xgroups, Ygroups):
            statspartial = functools.partial(statsfunction, X, Y, **kwargs)
            results.append([statspartial()])
        return results
    
    
    def testfunc(self, a, b, d=1, e=2):
        print a, b, d, e
        return True
    
    
    def testpart(self):
        xgroup = [1,2,3]
        ygroup = [3,2,1]
        results = self.sf_cv(self.testfunc, xgroup, ygroup, d='hi', e='hey')
        print results
        
        
    def convert_Y_vals(self, Y, original_val, new_val):
        replace = lambda val: new_val if (val==original_val) else val
        return [replace(v) for v in Y]
        
        
    def argtest(self, arg1, arg2, arg3, arg4, arg5):
        print arg1, arg2, arg3, arg4, arg5
        
    def putargs(self):
        args = [1,0,1,0,1]
        self.argtest(*[bool(x) for x in args])
    
if __name__ == '__main__':
    t = Tester()
    t.putargs()
    
    #testpart()
    #Y = [1,2,3,4,5,6,7,6,5,4,3,2,1,1,1]
    #print Y
    #print convert_Y_vals(Y, 1, 9)