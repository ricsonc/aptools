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
import config

if __name__ == '__main__':
    cmd = sys.argv[1]

    work_dir = sys.argv[2]

    if cmd == 'setup':
        for subdir in ['dark', 'raw', 'detections', 'warped', 'stacked', 'registration', 'demosaic']:
            ensure(f'{work_dir}/{subdir}')
            
    elif cmd == 'dark':
        utils.prepare_dark(files_in(work_dir, 'dark'))

    elif cmd == 'flat':
        utils.generate_flat(files_in(work_dir, 'flats'))

    elif cmd == 'demosaic':

        # for x in files_in(work_dir, 'raw'):
        #     print(x)
        #     utils.load_raw(x)
        
        from pathos.pools import ProcessPool 
        pool = ProcessPool(nodes=config.cores//2)
        pool.map( discard(utils.load_raw), files_in(work_dir, 'raw') )
        #need the `discard` to avoid leaking memory
        
    elif cmd == 'detect':
        det = Detector(work_dir, params=config.detection_params)
        from pathos.pools import ProcessPool 
        pool = ProcessPool(nodes=config.cores)
        pool.map( discard(det), files_in(work_dir, 'raw') )
        # for x in files_in(work_dir, 'raw'):
        #     det(x)
        
    elif cmd == 'register':
        Register(work_dir, params = config.registration_params)(files_in(work_dir, 'detections'))
 
    elif cmd == 'relregister':

        other_work_dir = sys.argv[3]
        print(f'registering relative to {other_work_dir}')
        Register(work_dir, params = config.registration_params)(files_in(work_dir, 'detections'), other=files_in(other_work_dir, 'detections'))
       
    elif cmd == 'warp':
        warper = Warp(work_dir, params = config.warping_params)

        from pathos.pools import ProcessPool 
        pool = ProcessPool(nodes=config.cores//2)
        pool.map( discard(warper), files_in(work_dir, 'raw') )
        
        #smap(warper, tqdm(files_in(work_dir, 'raw')))
        
    elif cmd == 'stack':
        Stacker(work_dir, params = config.stacking_params)(files_in(work_dir, 'warped', 'npy'))

    elif cmd == 'post':
        Postprocessor(work_dir, params = config.postprocess_params)(f'{work_dir}/stacked/out.tiff')
        
    else:
        raise Exception("unknown command")
