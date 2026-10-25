"""Full-epoch station TFT training; test evaluation is deliberately unavailable."""
import argparse
import json
from pathlib import Path
import tempfile

from citypulse_ai.forecasting.tft_training import train
from citypulse_ai.etl.archives import fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--max-seconds',type=int,default=10800);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--max-epochs',type=int,default=10);p.add_argument('--resume',type=Path)
    p.add_argument('--additional-epochs',type=int,help='Session cap; preserves original max-epochs and optimizer settings')
    p.add_argument('--gcs-prefix',help='Trusted internal artifact prefix in the existing CityPulse bucket')
    args=p.parse_args();publish=None
    if args.gcs_prefix:
        from citypulse_ai.cloud.gcs import GCSStorage
        from citypulse_ai.cloud.publish import publish_verified_artifact
        args.output_dir.mkdir(parents=True,exist_ok=True)
        verification=args.output_dir/'verification';verification.mkdir(exist_ok=True);tempfile.tempdir=str(verification)
        store=GCSStorage('citypulse-ai-511315','citypulse-ai-511315-data',upload_timeout=600)
        def publish(path):
            key=args.gcs_prefix.rstrip('/')+'/'+path.name
            return {'cloud_path':'gs://'+store.bucket_name+'/'+key,**publish_verified_artifact(path,key,store=store,
                metadata={'sha256':fingerprint(path),'artifact_role':'completed_epoch_tft_checkpoint'},
                max_bytes=32*1024**2,upload_attempts=1)}
    result=train(args.dataset,args.output_dir,max_seconds=args.max_seconds,batch_size=args.batch_size,
        max_epochs=args.max_epochs,resume=args.resume,publish=publish,additional_epochs=args.additional_epochs,progress=lambda s:print(s,flush=True))
    print(json.dumps({k:result.get(k) for k in ['status','epochs_completed','best','stop_reason','training_complete','validation']},indent=2))


if __name__=='__main__':main()
