usage: assign2.py [-h] [--train] -in INPUT -label LABEL
                  [-max_iter MAX_ITERATION] [-n_val NUM_VALIDATION]
                  [-lr LEARNING_RATE] [-bs BATCH_SIZE] [-reg REGULARIZATION]
                  [-p PATIENCE] [-pp_iter PRINT_PER_ITER] [-sw SAVE_WEIGHTS]
                  [-lw LOAD_WEIGHTS] [-pw] [-pl]

optional arguments:
  -h, --help            show this help message and exit
  --train               train flag
  -in INPUT, --input INPUT
                        path to the input data
  -label LABEL, --label LABEL
                        path to the label
  -max_iter MAX_ITERATION, --max_iteration MAX_ITERATION
                        max number of iterations (default 100)
  -n_val NUM_VALIDATION, --num_validation NUM_VALIDATION
                        validation set size (default 100)
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate (default 1e-3)
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size (default 50)
  -reg REGULARIZATION, --regularization REGULARIZATION
                        regularization factor (default 1e-4)
  -p PATIENCE, --patience PATIENCE
                        early stopping patience (default 10)
  -pp_iter PRINT_PER_ITER, --print_per_iter PRINT_PER_ITER
                        print per iteration (default 10)
  -sw SAVE_WEIGHTS, --save_weights SAVE_WEIGHTS
                        path to the output weights (default weights.p)
  -lw LOAD_WEIGHTS, --load_weights LOAD_WEIGHTS
                        path to the loaded weights
  -pw, --plot_weights   flag: plot weights
  -pl, --plot_loss      flag: plot loss