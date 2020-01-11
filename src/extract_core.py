import chess.pgn as pgn
import bz2
from ipdb import set_trace as st
from time import time
from multiprocessing import Pool, Queue
import sys

# def read_many_games(foo):
#     handle, tid, nt, q = foo
#     N = 100
    
#     for i in range(tid):
#         pgn.skip_game(handle)
    
#     for _ in range(N):
#         g = pgn.read_game(f)
#         if g is None:
#             break
#         q.put(g)

#         for i in range(nt):
#             pgn.skip_game(handle)

def read_game(g):
    head = g.headers
    tc = head['TimeControl']

    if head['Termination'] == 'Unterminated' or head['Result'] == '*':
        return False
    
    if '+' in tc:
        t0, t1 = tc.split('+')
        t = int(t0) + 40 * int(t1)
    elif tc == '-':
        t = -1

    try:
        elo = (int(head['WhiteElo']), int(head['BlackElo']))
    except:
        elo = (0,0)
        
    uci = [x.uci() for x in g.mainline_moves()]
    res = 2-int(head['Result'][-1]) #2 -> w, 1 -> b, 0 -> tie

    if ((min(elo) < 1510) or (t < 300)) and (min(elo) < 2000):
        return False
    elif (len(uci) < 10) or (len(uci) > 200):
        return False

    rank = 'A' if min(elo) >= 2000 and (t >= 300) else 'B'
    
    uci.append(str(res))
    
    return rank + ' ' + ' '.join(uci)

def redump(fn, fn2):
    f = bz2.open(fn, 'rt')
    f2 = open(fn2, 'w')
    
    i = 0
    j = 0

    t0 = time()
    while 1:
        try:
            g = pgn.read_game(f)
        except Exception as e:
            print('got error')
            print(e)
            continue
            
        if g is None:
            break

        if j % 100000 == 0:
            t1 = time()            
            print('%d/%d games in ' % (i,j), t1-t0)
            t0 = t1

        j += 1

        try:
            out = read_game(g)
        except Exception as e:
            print('got error')
            print(e)
            continue

        if not out:
            continue
            
        f2.write(out+'\n')

        i += 1

if __name__ == '__main__':
    name = sys.argv[1]
    name_out = name.split('.')[0]
    redump('chessgames/'+name, 'dump/'+name_out)
    print(name, 'done!')

    #ls -1 chessgames | grep bz2 | xargs -n 1 -P 8 python3 extract_core.py
