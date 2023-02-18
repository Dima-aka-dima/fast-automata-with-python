import numpy as np
import cv2 as cv
import time
import json

hardware = "GPU"

N      = 1024
M      = 1024
nSteps = 8192

FPS = 50.0
codec = "FFV1"
fourcc = cv.VideoWriter_fourcc(*codec)
videoFormat = 'avi'

colorAlive = "#000000"
colorDead  = "#FFFFFF"
# colorAlive = "#E06863"
# colorDead  = "#71FFA6"

if hardware == 'CPU':
    from numpy.lib.stride_tricks import as_strided, sliding_window_view
    lib = np
if hardware == 'GPU': 
    import cupy as cp
    from cupy.lib.stride_tricks import as_strided
    lib = cp
    from cupy.lib.stride_tricks import as_strided

filt = lib.array([[1,1,1],
                  [1,0,1],
                  [1,1,1]], dtype = np.uint8)

def init():
    allCells = lib.empty((nSteps + 1, N, M), dtype = lib.uint8)
    allCells[0] = lib.zeros((N,N))
    
    return allCells

def step(cells):
    # windows = sliding_window_view(cells, (3,3))
    windows = as_strided(cells, (N-2, M-2, 3, 3), (M, 1, M, 1))
    
    neigb = lib.einsum('ijkl,kl->ij', windows, filt)
    
    toAlive = lib.where((neigb == 3) & (cells[1:-1, 1:-1] == 0))
    toDead  = lib.where(((neigb < 2) | (neigb > 3)) & (cells[1:-1, 1:-1] == 1))
    
    cells[1:-1,1:-1][toAlive] = 1
    cells[1:-1,1:-1][toDead]  = 0
    
    return cells

def save():
    hex2bgr = lambda c: [int(c[i:i+2], 16) for i in (4, 2, 1)]

    colorAliveBGR = hex2bgr(colorAlive)
    colorDeadBGR  = hex2bgr(colorDead)
    
    video = cv.VideoWriter(f"run_{N}_{M}_{nSteps}_{hardware}.{videoFormat}",
                           fourcc, FPS, (N, M), True)

    if hardware == 'GPU': frames = allCells.get()
    if hardware == 'CPU': frames = allCells
    
    for frame in frames:
        image = frame.repeat(3).reshape(N, N, 3).astype(np.int16)
        image[image == 0] = -1
        
        for c in range(3):
            image[:,:,c][image[:,:,c] ==  1] = colorAliveBGR[c]
            image[:,:,c][image[:,:,c] == -1] = colorDeadBGR[c]
        
        video.write(image.astype(np.uint8))
    video.release()

def placePattern(patternName, where = 'center'):
    pattern = patterns[patternName]
    p = lib.array(pattern['pattern'], dtype = np.uint8)
    w, h = pattern['size']
    if type(where) == str:
        if where == 'center': allCells[0][N//2:N//2 + h, N//2:N//2 + w] = p
        if where == 'lefttop': allCells[0][2:h+2, 2:w+2] = p
    else:
        allCells[0][where[0]:where[0] + h, where[1]: where[1] + w] = p

filePatterns = open('patterns.json', 'r')
patterns = json.load(filePatterns)

startTime = time.time()

allCells = init()
placePattern('Jaydot', (N//3, N//3))
placePattern('Jaydot', (2*N//3, 2*N//3))

for i in range(1, nSteps + 1):
    allCells[i] = step(allCells[i-1].copy())

endTime = time.time()
print(f'Runtime: {endTime - startTime :.2f} s')

startTime = time.time()
save()
endTime = time.time()
print(f'Savetime: {endTime - startTime :.2f} s')
