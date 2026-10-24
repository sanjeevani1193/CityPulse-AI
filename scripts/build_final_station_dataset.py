"""Build the final fixed 100-station dataset from verified local historical inputs."""
import argparse
from pathlib import Path

from citypulse_ai.features.final_station_dataset import build_final_dataset
from citypulse_ai.features.station_history import StationHistory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--february-certificate',type=Path,required=True)
    p.add_argument('--certificate-sha256',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    history=StationHistory(args.audit,args.manifest,csv_inventories={
        '2025-02':(args.february_certificate,args.certificate_sha256)})
    result=build_final_dataset(history,args.output_dir,progress=lambda s:print(s,flush=True))
    print(result['split_statistics'],flush=True)


if __name__=='__main__':main()
