import argparse 
def parse_args():
    parser= argparse.ArgumentParser(description="training pipeline for credit scoring model")
    parser.add_argument(
        "--config", 
        type=str,
        required=True,
        help="Ruta al archivo YAML de configuracion"
    )
    
    return parser.parse_args()

