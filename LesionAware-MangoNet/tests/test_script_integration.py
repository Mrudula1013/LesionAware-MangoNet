import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def run_script(name,*args):
    proc=subprocess.run([sys.executable,str(ROOT/'scripts'/name),*map(str,args)],
                        cwd=ROOT,env={'PATH':'/usr/bin:/bin','PYTHONPATH':str(ROOT/'src'),
                                      'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},
                        capture_output=True,text=True,timeout=110)
    assert proc.returncode==0,f'{name} failed:\nstdout={proc.stdout}\nstderr={proc.stderr}'
    return proc.stdout


def test_demo_to_checkpoint_to_field_evaluation(tmp_path):
    data=tmp_path/'data'; runs=tmp_path/'runs'; evaluation=tmp_path/'evaluation'
    run_script('make_demo_data.py','--data-root',data,'--per-class','4')
    run_script('prepare_data.py','--data-root',data)
    run_script('train.py','--data-root',data,'--source','MangoLeafBD',
               '--seeds','0','--model-config',ROOT/'configs/tiny_model.yaml',
               '--training-config',ROOT/'configs/training.yaml',
               '--epochs','1','--batch-size','4','--num-workers','0',
               '--max-train-batches','1','--out',runs)
    ckpt=runs/'MangoLeafBD'/'seed_0'/'best.pt'
    assert ckpt.exists()
    run_script('evaluate.py','--checkpoint',ckpt,'--data-root',data,
               '--target','MLD-Field','--batch-size','4','--max-images','4','--out',evaluation)
    assert (evaluation/'predictions.csv').exists()
    assert (evaluation/'metrics.json').exists()
