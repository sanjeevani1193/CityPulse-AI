"""Station-level benchmark CLI; resource guards are applied by the VM wrapper."""
import argparse,json
from pathlib import Path
from citypulse_ai.forecasting.station_xgboost import train

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--tft-reference',type=Path,required=True)
    parser.add_argument('--max-seconds',type=int,default=9000)
    args=parser.parse_args()
    result=train(args.dataset,args.output_dir,max_seconds=args.max_seconds,reference=args.tft_reference,progress=lambda s:print(s,flush=True))
    print(json.dumps({k:result[k] for k in ['status','selected','model_bytes','elapsed_seconds','test_evaluated']},indent=2))
