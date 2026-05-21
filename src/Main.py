from utils.miscelaneous import turn_off_warnings
turn_off_warnings()

from utils import save_numpy, Multiprocess
import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from omegaconf import DictConfig
import time

#when running locally I use the local config file 

@hydra.main(version_base=None, config_path="../config", config_name="config")
# @hydra.main(version_base=None, config_name="config")
def main(cfg: DictConfig):

    inst = cfg.instance
    time_start = time.time()
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    
    worker = Multiprocess(inst, run_dir)
    results = worker.run()
    

    
    print("Saving results to:", run_dir / "results.npy")
    
    save_numpy(str(run_dir / "results.npy"), results)

    time_end = time.time()
    
    #write out how long time the job took, hh:mm:ss
    
    elapsed = int(time_end - time_start)

    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Job finished in: {hours} hours, {minutes} minutes, and {seconds} seconds")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()