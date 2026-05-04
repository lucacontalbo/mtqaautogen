from models import MTAutoGen
from utils import get_args
import os
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    args = get_args()
    mtautogen = MTAutoGen(args)
    method, domain, num_tables, generate_ablations = args['question_type'], args['domain'], args['num_tables'], args['generate_ablations']
    if num_tables == 1:
        num_table_str = "one-table"
    elif args["sequential"]:
        num_table_str = "multi-table-sequential"
    else:
        num_table_str = "multi-table"

    if domain is None:
        domain = "general_domain"

    path = os.path.join("datasets", num_table_str)
    os.makedirs(path, exist_ok=True)

    if generate_ablations:
        samples, error_logs = mtautogen.run_ablations(num_tables=num_tables, num_samples=50, domain=domain, sequential=args["sequential"])
        print(error_logs)
        for num_tables in samples:
            for k1 in samples[str(num_tables)]:
                for k2 in samples[str(num_tables)][k1]:
                    path_k1_k2 = os.path.join(path, str(num_tables), domain, k1, k2, "data.csv")
                    os.makedirs(os.path.dirname(path_k1_k2), exist_ok=True)
                    samples[str(num_tables)][k1][k2].to_csv(path_k1_k2, index=False)
    else:
        samples = mtautogen.run_loop(method=method, num_samples=50, domain=domain, num_tables=num_tables, sequential=args["sequential"])
        path = os.path.join(path, str(num_tables), domain, method+".csv")
        samples.to_csv(path, index=False)
