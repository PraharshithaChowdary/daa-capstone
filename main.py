import argparse
import logging
import sys
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Fraud Pattern Detection using Graph Mining'
    )
    parser.add_argument(
        '--mode', type=str, choices=['train', 'predict', 'api', 'demo', 'visualize'],
        default='demo', help='Operation mode'
    )
    parser.add_argument(
        '--data', type=str, help='Path to transaction data'
    )
    parser.add_argument(
        '--model', type=str, help='Path to saved model'
    )
    parser.add_argument(
        '--output', type=str, default='output', help='Output directory'
    )
    parser.add_argument(
        '--host', type=str, default='0.0.0.0', help='API host'
    )
    parser.add_argument(
        '--port', type=int, default=8000, help='API port'
    )
    parser.add_argument(
        '--generate-data', action='store_true', help='Generate synthetic data'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        from pipeline import run_demo
        pipeline = run_demo()
        
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            pipeline.save_model(os.path.join(args.output, 'fraud_detector.pkl'))
            logger.info(f"Results saved to {args.output}/")
    
    elif args.mode == 'train':
        from pipeline import FraudDetectionPipeline
        pipeline = FraudDetectionPipeline()
        
        if args.generate_data:
            pipeline.generate_synthetic_data()
        elif args.data:
            pipeline.load_data(args.data)
        else:
            logger.error("Either --data or --generate-data is required for train mode")
            sys.exit(1)
        
        pipeline.extract_graph_features()
        pipeline.prepare_labels()
        pipeline.train_model()
        
        os.makedirs(args.output, exist_ok=True)
        pipeline.save_model(os.path.join(args.output, 'fraud_detector.pkl'))
        
        summary = pipeline.get_summary()
        
        with open(os.path.join(args.output, 'report.json'), 'w') as f:
            import json
            json.dump(summary, f, indent=2, default=str)
        
        patterns = pipeline.get_suspicious_patterns()
        with open(os.path.join(args.output, 'suspicious_patterns.json'), 'w') as f:
            json.dump(patterns, f, indent=2, default=str)
        
        print(f"\nModel and report saved to {args.output}/")
    
    elif args.mode == 'api':
        from api.server import run_server
        run_server(host=args.host, port=args.port)
    
    elif args.mode == 'predict':
        if not args.model or not args.data:
            logger.error("Both --model and --data are required for predict mode")
            sys.exit(1)
        
        from pipeline import FraudDetectionPipeline
        from graph.transaction_graph import Transaction
        import pandas as pd
        from datetime import datetime
        
        pipeline = FraudDetectionPipeline()
        pipeline.load_model(args.model)
        pipeline.load_data(args.data)
        pipeline.extract_graph_features()
        
        predictions = []
        for _, tx in pipeline.raw_data.iterrows():
            transaction = Transaction(
                transaction_id=str(tx.get('transaction_id', '')),
                sender_id=str(tx['sender_id']),
                receiver_id=str(tx['receiver_id']),
                amount=float(tx['amount']),
                timestamp=pd.to_datetime(tx.get('timestamp', datetime.now()))
            )
            result = pipeline.predict(transaction)
            predictions.append({
                'transaction_id': transaction.transaction_id,
                **result
            })
        
        results_df = pd.DataFrame(predictions)
        results_df.to_csv(os.path.join(args.output, 'predictions.csv'), index=False)
        print(f"Predictions saved to {os.path.join(args.output, 'predictions.csv')}")
    
    elif args.mode == 'visualize':
        if not args.model or not args.data:
            logger.error("Both --model and --data are required for visualize mode")
            sys.exit(1)
        
        from pipeline import FraudDetectionPipeline
        from visualization.graph_viz import create_interactive_report
        
        pipeline = FraudDetectionPipeline()
        pipeline.load_model(args.model)
        pipeline.load_data(args.data)
        pipeline.extract_graph_features()
        
        os.makedirs(args.output, exist_ok=True)
        report_path = os.path.join(args.output, 'fraud_report.html')
        create_interactive_report(
            pipeline.graph.graph,
            pipeline.mining_results,
            report_path
        )
        print(f"Visual report saved to {report_path}")


if __name__ == "__main__":
    main()
