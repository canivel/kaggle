# === subprocess_worker.py ===
import time
import_timer = time.time()
import sys
import torch
import pickle
from common_stuff import prepare_dataset, Decoder, arc_test_set, prepare_run, aug_score_params, score_temp_storage, keep_max
from model_runner import inference_step, process_inference_output, mem_info, disk_info
from pp import *
from tqdm import tqdm
if pp:print('import time',round(time.time()-import_timer))

if __name__ == "__main__":
    base_key, storage_path, gpu, out_path, max_new_tokens, min_prob, min_prob_greedy, temperature, index, store, key_separated = sys.argv[1:]

    gpu = int(gpu)
    # max_new_tokens = int(max_new_tokens)
    min_prob = float(min_prob)
    min_prob_greedy = float(min_prob_greedy)
    temperature = float(temperature)
    i = int(index)
    if store == 'None':
        store = None
    key_separated = bool(int(key_separated))

    if pp:mem_info(f'{i}.before_model_'+base_key)
    start_timer = time.time()
    model, formatter = prepare_run(storage_path, gpu=gpu, inf=True)
    prepare_run_fin = time.time()
    prepare_run_time = prepare_run_fin-start_timer
    if pp:
        print(i,'prepare_run_time',round(prepare_run_time))
        mem_info(f'{i}.after_model_'+base_key)
    if max_new_tokens == "None":
        max_new_tokens = formatter.max_new_tokens()
    else:
        max_new_tokens = int(max_new_tokens)

    if pp:print('max_new_tokens',max_new_tokens)

    dataset = prepare_dataset(formatter, train=False, gpu=gpu, key=base_key.split('_')[0])
    base_keys = []
    for key in dataset.keys:
        if key.startswith(base_key):
            base_keys.append(key)

    if len(base_keys)==len(dataset.keys) and key_separated:
        # base_keys = base_keys[::-i]
        base_keys = base_keys[i::2]
    if pp:print(i,'base_keys len:',len(base_keys),base_keys)
    decoder = Decoder(formatter, arc_test_set.keep_key_startswith(base_key.split('_')[0]).split_multi_replies(), n_guesses=2, prob_baseline=0.05, keep_max=keep_max)
    # if pp:print('probtracker_best',decoder.prob_tracker_best)
    # if pp:mem_info(f'{i}.after_decoder_'+base_key)

    formatter.tokenizer.padding_side = 'left'
    result_dict = {}

    data_decoder_fin = time.time()

    cb_time = 0
    for key in tqdm(base_keys, file=sys.stdout, desc=f'sp_inf{i} {base_key}') if pp else base_keys:
        # if pp:print(i,'\n')
        input_text = dataset.get(key, formatter)['input']
        batch = formatter.tokenizer([input_text], return_tensors='pt')

        cb_start = time.time()
        current_best = decoder.get_current_best(key.split('.')[0],batch=key_separated)
        cb_time += time.time()-cb_start
        if current_best is not None:
            # if pp:print(i,'current_best',current_best[0])
            current_best = dataset.forward_mod(current_best, key)
            current_best = formatter.fmt_reply([current_best])
            current_best = formatter.tokenizer(input_text + current_best)['input_ids'][batch['input_ids'].shape[-1]:]

        batch_out = inference_step(
            batch, model, formatter=formatter, max_new_tokens=max_new_tokens,
            current_best=current_best, min_prob=min_prob, min_prob_greedy=min_prob_greedy, temperature=temperature
        )
        outputs = [x[0] for x in batch_out]

        # if pp:print('\noutputs',outputs)
        result_dict[key] = process_inference_output(
            key, outputs, formatter, decoder=decoder, store=store, decoder_args={'batch':key_separated}
        )

    inf_fin = time.time()
    inf_time = inf_fin - data_decoder_fin
    if pp:print(i,'inf_time',round(inf_time),'key_separated',key_separated,'cb_time',cb_time)

    if use_aug:decoder.calc_augmented_scores(model=model, store=score_temp_storage, quiet=not pp, **aug_score_params)
    calc_fin = time.time()
    calc_time = calc_fin - inf_fin
    if pp:print(i,'calc_time',round(calc_time))

    result_dict[f'mem{i}'],_ = mem_info(f'{i}.subprocess_completed_'+base_key)
    with open(out_path, 'wb') as f:
        pickle.dump(result_dict, f)

    print(f"*** Subprocess for {base_key} completed.",round(time.time()-start_timer),'s')

