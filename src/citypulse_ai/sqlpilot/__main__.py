"""Operator CLI. Never enables a model/API or downloads anything implicitly."""
import argparse
import json
from citypulse_ai.sqlpilot.core import SafeDatabase, QueryRejected
from citypulse_ai.sqlpilot.agent import SQLPilot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',required=True)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--schema',action='store_true')
    choice.add_argument('--sql')
    choice.add_argument('--question')
    choice.add_argument('--evaluate',action='store_true')
    args=parser.parse_args();database=SafeDatabase(args.database)
    try:
        if args.schema: result=database.discover()
        elif args.sql: result=database.query(args.sql)
        elif args.question: result=SQLPilot(database).ask(args.question)
        else:
            from citypulse_ai.sqlpilot.evaluation import evaluate
            result=evaluate(args.database)
        print(json.dumps(result,indent=2,allow_nan=False))
        if args.evaluate and (result['correct_results'] != result['total_cases'] or result['unsafe_rejected'] != result['unsafe_total']):
            raise SystemExit(1)
    except QueryRejected as error:
        print(json.dumps({'error':str(error)}));raise SystemExit(1)


if __name__=='__main__':main()
