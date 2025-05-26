# Find file and print out the file 

import os
import sys

def find_file_and_print(path, filename):
    for root, dirs, files in os.walk(path):
        if filename in files:
            with open(os.path.join(root, filename), 'r') as f:
                print(f.read())
            return True
    return False
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python see.py <filename>')
        sys.exit(1)
    find_file_and_print(f'results_WT/tofu/llama2-7b/{sys.argv[1]}/{sys.argv[2]}/seed_1001/epoch{sys.argv[3]}_{sys.argv[4]}_FixRefFalse_maskTrue_1.0_1.0/1/unlearn_times_1/eval_results-last', 'unlearning_results.txt')
    find_file_and_print(f'results_WT/tofu/llama2-7b/{sys.argv[1]}/{sys.argv[2]}/seed_1001/epoch{sys.argv[3]}_{sys.argv[4]}_FixRefFalse_maskTrue_1.0_1.0/1/unlearn_times_1/eval_results-last', 'mixed_results.txt')