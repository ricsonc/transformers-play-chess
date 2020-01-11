import itertools as it

def get_moves():
    
    def totext((x,y)):
        def totext2((u,v)):
            return 'abcdefgh'[u]+str(v+1) 
        def totext3((u,v,k)):
            return 'abcdefgh'[u]+str(v+1)+'rbnq'[k]
        def totext23(x):
            return (totext2 if len(x) == 2 else totext3)(x)
        return totext23(x)+totext23(y)
        
    def nbh((u,v)):
        for i in range(8):
            yield (i,v)
            yield (u,i)

            yield (i,v+u-i)
            yield (i,v-u+i)

        for m1, m2 in it.product([-1,1],[-1,1]):
            yield (u+m1, v+2*m2)
            yield (u+2*m1, v+m2)

        for k in range(4):
            for w in [u-1,u,u+1]:
                if v == 6:
                    yield (w,7,k)
                elif v == 1:
                    yield(w,0,k)
        
    def is_valid(x, y):
        u,v = y[:2]
        if x == y:
            return False
        if u < 0 or v < 0 or u >= 8 or v >= 8:
            return False
        return True

    valid = []
    for x in it.product(range(8), range(8)):
        valid.extend( [(x,y) for y in nbh(x) if is_valid(x, y)] )
        
    valid = set(valid)
    return sorted(list(map(totext, valid)) + list('012AB'))

for x in get_moves():
    print x


