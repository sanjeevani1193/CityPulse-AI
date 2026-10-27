"""Station LSTM training/validation only; infrastructure guards are caller-owned."""
import argparse
import json
from pathlib import Path
from citypulse_ai.forecasting.station_lstm import probe, train

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--resume')
    parser.add_argument('--max-seconds',type=int,default=10000)
    args=parser.parse_args()
    if args.probe:
        result=probe(args.dataset,args.output_dir)
    else:
        if args.reference is None:parser.error('--reference required for training')
        result=train(args.dataset,args.output_dir,args.reference,max_seconds=args.max_seconds,
                     resume=args.resume,progress=lambda stage:print(stage,flush=True))
    print(json.dumps(result,indent=2))
