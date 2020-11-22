import sys
from ipdb import set_trace as st

from star_detect import Detector
from registration import Register
from warping import Warp
from stack import Stacker
from postprocess import Postprocessor
import utils

from common import *
from munch import Munch as M
from tqdm import tqdm

if __name__ == '__main__':
    cmd = sys.argv[1]

    # work_dir = 'tmp_data'
    # work_dir = 'andromeda'
    # work_dir = 'andro2'
    # work_dir = 'triangulum'
    # work_dir = 'hatri'
    #work_dir = 'haorion'
    #work_dir='bloop'
    work_dir='oneb'

    if cmd == 'setup':
        for subdir in ['dark', 'raw', 'detections', 'warped', 'stacked', 'registration', 'demosaic']:
            ensure(f'{work_dir}/{subdir}')
    elif cmd == 'dark':
        utils.prepare_dark(files_in(work_dir, 'dark'))
    elif cmd == 'demosaic':

        from pathos.pools import ProcessPool 
        pool = ProcessPool(nodes=4)
        pool.map( discard(utils.load_raw), files_in(work_dir, 'raw') )
        #need the `discard` to avoid leaking memory
        
    elif cmd == 'detect':
        det = Detector(work_dir)
        smap(det, files_in(work_dir, 'raw'))
    elif cmd == 'register':
        Register(work_dir)(files_in(work_dir, 'detections'))
    elif cmd == 'warp':
        warper = Warp(work_dir)

        from pathos.pools import ProcessPool 
        pool = ProcessPool(nodes=4)
        pool.map( discard(warper), files_in(work_dir, 'raw') )
        
        #smap(warper, tqdm(files_in(work_dir, 'raw')))
        
    elif cmd == 'stack':
        Stacker(work_dir)(files_in(work_dir, 'warped', 'npy'))
    elif cmd == 'post':
        Postprocessor(work_dir)(files_in(work_dir, 'stacked', 'tiff')[0])
    else:
        raise Exception("unknown command")
