import numpy as np
import itertools
import time
from PIL import Image

hardware = "CPU"

N      = 1024
nSteps = 512
nRule  = 30
start = "middle"

colorAlive = "#E06863"
colorDead  = "#71FFA6"

fileFormat = "png"

if hardware == 'CPU':
    from numpy.lib.stride_tricks import as_strided 
    lib = np

if hardware == 'GPU': 
    import cupy as cp
    from cupy.lib.stride_tricks import as_strided
    lib = cp
    from cupy.lib.stride_tricks import as_strided

mapToArray = lambda f, xs: np.array(list(map(f, xs)))

formatPattern = lambda p: 2*mapToArray(int,p)-1

def getRules():    

    patterns = mapToArray(''.join, list(itertools.product("01", repeat = 3)))[::-1]
    
    binary = np.binary_repr(nRule)
    binary = np.array([*('0'*((1<<3)-len(binary)) + binary)])
    
    zeros = np.where(binary == '0')[0]
    ones  = np.where(binary == '1')[0]
    
    ruleToDead  = mapToArray(formatPattern, patterns[zeros])
    ruleToAlive = mapToArray(formatPattern, patterns[ones] )
    
    return lib.array(ruleToDead ),\
           lib.array(ruleToAlive)

def init():

    allCells = lib.empty((nSteps + 1, N), dtype = lib.int8)
    
    if start == 'random': allCells[0] = 2*lib.random.randint(0, 2, N) - 1
    else:                 allCells[0] = -lib.ones(N)
    if start == 'middle': allCells[0][N//2] = 1
    if start == 'left'  : allCells[0][0]    = 1
    if start == 'right' : allCells[0][N-1]  = 1
    
    return allCells

def step(cells):
    
    # Creates sliding windows without additional memory
    # For numpy implementation can use sliding_window_view
    windows = lib.lib.stride_tricks.as_strided(cells, (cells.shape[0] - 2, 3), (1, 1))
    
    # Cells are -1 for Dead and 1 for Alive
    # Which means convolution with the fitting rule gives 3
    # One pass is actually enough, but the performance doesn't drop
    toDead  = lib.where((lib.einsum('ij,kj->ik', windows, rules[0]) == 3).any(1))[0] + 1
    toAlive = lib.where((lib.einsum('ij,kj->ik', windows, rules[1]) == 3).any(1))[0] + 1
    
    cells[toAlive] =  1
    cells[toDead]  = -1 
    return cells

def save():
    hex2rgb = lambda c: [int(c[i:i+2], 16) for i in (1, 2, 4)]
    
    colorAliveRGB = hex2rgb(colorAlive)
    colorDeadRGB  = hex2rgb(colorDead)
    
    if hardware == 'GPU': image = allCells.get()
    if hardware == 'CPU': image = allCells
        
    image = image.repeat(3).reshape(nSteps + 1, N, 3).astype(np.int16)
    for i in range(3):
        image[:,:,i][image[:,:,i] == 1]  = colorAliveRGB[i]
        image[:,:,i][image[:,:,i] == -1] = colorDeadRGB[i]
    
    image = Image.fromarray(image.astype(np.uint8))
    image.save(f"run_{N}_{nSteps}_rule{nRule}_{start}_{hardware}.{fileFormat}")

    
rules = getRules()

startTime = time.time()

allCells = init()

for i in range(1, nSteps + 1):
    allCells[i] = step(allCells[i-1].copy())

endTime = time.time()

save()

print(f"Runtime: {endTime - startTime :.2f} s")
