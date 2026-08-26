# common configuration for training and evaluation
from pp import *
from arc_loader import *
from model_runner import *
from selection import *
from async_tools import *
import time
import os
import torch
import random
import numpy as np

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using multi-GPU

# For reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import traceback
gpu_count = torch.cuda.device_count()
print('gpu_count',gpu_count)
import warnings
warnings.filterwarnings("ignore")

# paths
tmp_dir = '/kaggle/temp'
# tmp_dir = '/kaggle/working'
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    arc_challenge_file = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
else:
    arc_challenge_file = '/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json'
arc_solutions_file = '/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json'
model_temp_storage = os.path.join('', 'finetuned_model')
infer_temp_storage = os.path.join(tmp_dir, 'inference_outputs')
score_temp_storage = os.path.join(tmp_dir, 'inference_scoring')

keep_model = False
do_train = False
get_diff = False
one_rotp = False
one_ex = False
one_ex_mix = False
keep_max = 'rand' if one_ex_mix else None
train_classifier = False
if train_classifier:
    do_train = True
    arc_challenge_file = arc_challenge_file.replace('evaluation','training')
    arc_solutions_file = arc_solutions_file.replace('evaluation','training')
use_classifier = c_head

# load datasets
arc_test_set = ArcDataset.from_file(arc_challenge_file, start=300 if do_train else 0, limit_n=300 if do_train else 120)
obj_start_time = time.time()
if arc_test_set.is_fake: arc_test_set.load_replies(arc_solutions_file)
keys = arc_test_set.sorted_by_len(formatter=None, name='input').keys
new_keys = []
check_index = []
for n,k in enumerate(keys):
    if n%8<5:
        new_keys.append(k)
        check_index.append(n)
    else:
        new_keys.insert(4-n%8,k)
        check_index.insert(4-n%8,n)
# if pp:print('check_index',check_index)
keys = new_keys

if arc_test_set.is_fake and not train_classifier:
    keys = keys[1:2] + keys[8:9] + keys[14:15] + keys[23:24]
    arc_test_set_ = arc_test_set.change_keys(keys)
    obj_ds = arc_test_set_.make_object_dataset(make=False)
    bg_cut_ds = obj_ds.make_background_cut_dataset()
    for n,key in enumerate(keys):
        got_length = arc_test_set_.get_length(key,formatter=None,name='input')
        visualize_task(key,title=f"#{n} len input:{got_length}",file='eval')
        # print([v for k,v in obj_ds.background_values.items() if k[0]==key])
        # print([v for k,v in bg_cut_ds.background_values.items() if k[0]==key])
        try:
            if key in bg_cut_ds.keys:
                visualize_task(bg_cut_ds.queries[key],title=f"#{n} background_cut {key}",file='eval')
        except:
            print(bg_cut_ds.queries[key])
            print(traceback.format_exc())

# models
base_model, MyFormatter, perm_aug, max_seq_length_train, mask_first = '/kaggle/input/wb55l_nemomini_fulleval/transformers/default/1', ArcFormatter_premix_3 if pp else ArcFormatter_premix_3, 'rnd_all', 2000 if one_ex else 4224, 0

# training & inference
train_epochs = 24#4
multi_gpu_train = False if train_classifier else True
multi_gpu_random_split = True
max_seq_length_infer = 6336# 2000 if one_ex else 8192
prime_on_single_task = False
infer_params = dict(min_prob=0.17, store=infer_temp_storage, use_turbo=True)

# scoring
use_aug_score = use_aug
aug_score_params = dict(tp=True, rot=True, perm=perm_aug, shfl_ex=True, make_unique=False, max_len=max_seq_length_infer)
submission_select_algo = score_full_probmul_3 if use_aug_score else score_all_probsum

def prepare_run(model_path, load_lora=None, train=False, gpu=None, inf=False, **kwargs):
    if gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"   ] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if train_classifier:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    else:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'embed_tokens', 'lm_head']
        
    model, tokenizer, formatter = prepare_model(  # base model configuration
        model=model_path,
        local_files_only=True,
        mode='unsloth_4bit',
        # mode='transformers',
        #shrink_embedding=8000,
        max_seq_length=max_seq_length_train,
        formatter=MyFormatter,
        inf=inf,
        peft=([dict(
            r=64 if train_classifier else 64 if pp else 64,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=target_modules,
            lora_alpha=16 if train_classifier else 64 if pp else 16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            use_gradient_checkpointing=True,  # True or "unsloth" for very long context
            random_state=42,
            use_rslora=True,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )] if train or load_lora else []) + ([load_lora] if load_lora else []),
        **kwargs
    )
    # if gpu==0 and train and pp:print(model)
    
    if train and mask_first: formatter.collator_kwargs.update(mask_first_output=mask_first)

    return model, formatter

def prepare_dataset(formatter, train, gpu=None, key=None):
    ds = arc_test_set
    if key:
        # ds.keys = [key]
        ds = ds.change_keys([key])
        
    ds = ds.make_object_dataset(make=False)
    ds = ds.make_background_cut_dataset()
    if train:
        if do_train:
            ds = ds.move_test_to_train(only_when_2=True)
        ds = ds.remove_replies()
        if get_diff:ds = ds.get_diff()
        ds = ds.augment(tp=True, rot=True, perm=perm_aug, n=1 if do_train else 24 if arc_test_set.is_fake else train_epochs, shfl_ex=True, shfl_keys=True, esc=False, keep_max=keep_max)
        # if one_ex or one_rotp:ds = ds.cut_to_query_count(1)
        ds = ds.cut_to_query_count(cut_query+1)
        if cut_query>2:ds = ds.cut_to_query_count(cut_query,p=0.5)
        if one_rotp:ds = ds.one_rotp()
        # if pp:ds = ds.one_color().shuffled()
        # if pp:ds.keys =  ds.keys[:train_epochs*8]
        ds = ds.cut_to_len(formatter=formatter, name='text', max_len=max_seq_length_train, max_new_tokens=0)
        if pp:print('ds.keys in train prepare_dataset', len(ds.keys), ds.keys[:2])
        # if arc_test_set.is_fake: ds = ds.sorted_by_len(formatter=formatter, name='text', reverse=True)
    else:
        # ds = ds.sorted_by_len(formatter=formatter, name='input', max_of_transposed=True)
        ds = ds.split_multi_replies()
        if get_diff:ds = ds.get_diff()
        ds = ds.augment(tp=True, rot=True, n=1, seed=42, perm=perm_aug, shfl_ex=True, esc=False, keep_max=keep_max).interleave(ds.length())
        # if one_ex or one_rotp:ds = ds.cut_to_query_count(0)

        if use_ext:
            even_keys = [k for n, k in enumerate(ds.keys) if n % 2 == 0]
            odd_keys = [k for n, k in enumerate(ds.keys) if n % 2 != 0]
            
            ds_even = ds.change_keys(even_keys)
            ds_odd  = ds.change_keys(odd_keys)
            
            # Apply transformations to even part
            ds_even = ds_even.remove_replies()
            ds_even = ds_even.last_train_ex_for_test()
            ds_even = ds_even.split_multi_replies(train=True)
            
            # Combine the datasets again
            ds = ArcDataset.append(ds_even, ds_odd)

        ds = ds.cut_to_query_count(cut_query)
        if cut_query>2:ds = ds.cut_to_query_count(cut_query-1,p=0.5)
        if one_rotp:ds = ds.one_rotp()
        # if pp:ds = ds.one_color().shuffled()
        ds = ds.cut_to_len(formatter=formatter, name='input', max_len=max_seq_length_infer)

        # ds = ds.sorted_by_len(reverse=True, formatter=formatter, name='input')
        if pp:print('ds.keys in inference prepare_dataset',len(ds.keys), ds.keys[0:4],ds.keys[-4:])#,'\n-> 128')
        # if arc_test_set.is_fake: ds.keys = ds.keys[:4] #ds.keys[::-1][::5][::-1]

    return ds

def start_training(gpu):
    import gc
    import time
    import json
    import traceback
    time_limit = time.time() + 10.5*60*60

    stats = []
    for nkey,key in enumerate(keys[gpu::gpu_count]):

        if pp:print('\nnkey',nkey,'gpu',gpu,key)
        if time.time() > time_limit or nkey>0 and do_train:
            break
        try:
            train_start_time = time.time()
            storage_path = "Classifier_train_try65" if do_train else f'{model_temp_storage}_gpu{gpu}'
            if (gpu==0 or multi_gpu_train):# and not os.path.exists(storage_path):
                with RemapCudaOOM():
                    # if pp:mem_info('before prepare_run') #Don't get mem info before prepare_run. Raise error.
                    if nkey==0 or not keep_model:
                        use_lora = True
                        model, formatter = prepare_run(base_model, train=use_lora, gpu=gpu if multi_gpu_train else None)
                        # if use_classifier: model = load_classifier_head(model)
                    if pp:mem_info('after prepare_run')
                    prepare_run_fin = time.time()
                    prepare_run_time = prepare_run_fin-train_start_time
                    if pp:print('prepare_run_time',round(prepare_run_time))
                    dataset = prepare_dataset(formatter, train=True, gpu=gpu if multi_gpu_train else None, key=None if do_train else key)
                    t_dataset = prepare_dataset(formatter, train=False, gpu=gpu, key=key)
                    len_test = len(set([v.split('.')[0] for v in t_dataset.keys]))
                    data_len = arc_test_set.get_length(key=key,formatter=formatter,name='text')
                    if pp:
                        sub_formatter = ArcFormatter_premix_3(tokenizer=formatter.tokenizer)
                        data_len_long = arc_test_set.get_length(key=key,formatter=sub_formatter,name='text')
                    if pp:print('data_len',data_len,'data_len_long',data_len_long,'len_test',len_test)
                    # if nkey==gpu==0 and arc_test_set.is_fake:
                    #     if pp:print(arc_test_set.get(key,formatter))
                    if pp:print('data_time',round(time.time()-prepare_run_fin))
                    skip_train = False
                    if skip_train:
                        pass
                    elif train_classifier or use_classifier:
                        # model, loss_log, total_loss = classifier_train_with_accelerate(model, dataset, formatter.tokenizer, formatter, store=storage_path)
                        model, loss_log, total_loss = classifier_train_loop(model, dataset, formatter.tokenizer, formatter, store=storage_path)
                    else:
                        model, trainer_stats, trainer = training_run(
                            model, formatter, dataset, store=storage_path,
                            max_seq_length=max_seq_length_train,
                            grad_acc_fix=False,
                            train_args=dict(
                                per_device_train_batch_size=4 if pp else 4,
                                gradient_accumulation_steps=2 if pp else 2,
                                #from try13&14,b4a1s48 is faster loss drop and redundant inf outpu than b8a1s24, same tr time 147-156s for id0, 240-243s for id3
                                warmup_steps=4,
                                num_train_epochs=1,
                                #max_steps=20 if arc_test_set.is_fake else -1,
                                max_steps=-1,# if do_train else 24 if arc_test_set.is_fake else 24, #ins 20250329
                                learning_rate=1e-4,
                                embedding_learning_rate=1e-5,
                                logging_steps=4,
                                optim="adamw_8bit",
                                weight_decay=0.01,  # 0.01,
                                lr_scheduler_type='constant' if do_train else 'cosine',  # "linear", "cosine", "constant"
                                seed=42,
                                output_dir=os.path.join(tmp_dir, 'checkpoints'),
                                save_strategy="no",
                                report_to='none',
                            ),
                        )

                    train_mem_usage, _ = mem_info('train')

                    del dataset#, model, formatter
                    b_gc = time.time()
                    gc.collect()
                    train_run_fin = time.time()
                    if pp:print('gc time',round(train_run_fin-b_gc))
                    train_run_time = train_run_fin-prepare_run_fin
                    if pp:print('train_run_time', round(train_run_time))
                
                    inf_data_len = arc_test_set.get_length(key=key,name='text',formatter=formatter)
                    if pp:print('inf_data_len',inf_data_len, 'len_test',len_test)
                    retrainer = None if not prime_on_single_task else Retrainer(
                        n=32,
                        aug_opts=dict(perm=perm_aug, shfl_ex=True),
                        reload_state_dict=get_and_fix_peft_weights(storage_path),
                        formatter=formatter,
                        max_seq_length=max_seq_length_infer,
                        grad_acc_fix=False,
                        train_args=dict(
                            per_device_train_batch_size=2,
                            gradient_accumulation_steps=2,
                            warmup_steps=4,
                            num_train_epochs=1,
                            learning_rate=1e-4,
                            embedding_learning_rate=0,
                            max_steps=20 , #ins 20250329
                            logging_steps=8,
                            optim="adamw_8bit",
                            weight_decay=0.00,  # 0.01,
                            lr_scheduler_type='constant',  # "linear", "cosine",
                            seed=42,
                            output_dir='tmp_output',
                            save_strategy='no',
                            report_to='none',
                        ),
                    )
                    decoder = Decoder(formatter, arc_test_set.keep_key_startswith(key).split_multi_replies(), n_guesses=2, prob_baseline=0.05, keep_max=keep_max)
                    if 1:# gpu in [0,1,2,3] and len_test>0 and not pp or gpu in [0,1,2,3] and pp and not use_classifier:
                        import traceback
                        try:
                            if not keep_model: del model, formatter
                            gc.collect()
                            torch.cuda.empty_cache()

                            if pp:mem_info('before_inf_batch')
                            if pp:time.sleep(5)
                            batch_result = inference_run_batch(t_dataset, storage_path=storage_path, gpu=gpu, max_subprocesses=2 if gpu_count==4 else 1, **{k:v if k=='min_prob' and pp else v for k,v in infer_params.items()})
                            inf_mem_usage = batch_result['mem']
                            if not keep_model: model, formatter = None, None

                        except:
                            print('tried inference_run_batch',traceback.format_exc())
                            if not keep_model: model, formatter = prepare_run(storage_path, gpu=gpu)
                            inference_run_v2(model, formatter, t_dataset, decoder, retrain=retrainer, **infer_params)
                            inf_mem_usage,_ = mem_info('inf')

                    else:
                        inference_run_v2(model, formatter, t_dataset, decoder, classifier=True, retrain=retrainer, **infer_params)
                        inf_mem_usage,_ = mem_info('inf')
                    inf_run_fin = time.time()
                    inf_run_time = inf_run_fin-train_run_fin
                    if pp:print('inf_run_time',round(inf_run_time))
                    if use_aug_score or arc_test_set.is_fake: decoder.calc_augmented_scores(model=model, store=score_temp_storage, quiet=not pp, **aug_score_params)
                    calc_fin = time.time()
                    calc_time = calc_fin-inf_run_fin
                    if pp:print('calc_time',round(calc_time))
                    calc_mem_usage,_ = mem_info('calc')
                    
                    if not keep_model: del model, formatter, t_dataset, decoder
                    if os.path.exists(storage_path) and not do_train:
                        import shutil
                        shutil.rmtree(storage_path)
                    gc.collect()

            total_time = time.time()-train_start_time
            if pp:print('Train Inf nkey',nkey,'gpu',gpu,key, round(total_time),'s')
            if pp:print('  data_len',data_len)
            if pp:print('  prepare_run_time',round(prepare_run_time))
            if pp:print('  train_run_time', round(train_run_time))
            if pp:print('  train_mem_usage', train_mem_usage)
            if pp:print('  inf_data_len',inf_data_len)
            if pp:print('  inf_run_time',round(inf_run_time),'inf_run_time/n',round(inf_run_time/(len_test+0.001)))
            if pp:print('  inf_mem_usage',inf_mem_usage)
            if pp:print('  calc_time',round(calc_time))
            if pp:print('  calc_mem_usage',calc_mem_usage)
            if pp:
                if skip_train:
                    loss_log = {}
                    total_loss = 0
                
                elif not train_classifier and not use_classifier:
                    total_loss = trainer_stats.training_loss
                    loss_log = trainer.state.log_history
                stat = {
                    'nkey': nkey,
                    'gpu': str(gpu),
                    'key': key,
                    'total_time': total_time,
                    'data_len': data_len,
                    'data_len_long': data_len_long,
                    'prepare_run_time': prepare_run_time,
                    'train_run_time': train_run_time,
                    'train_mem_usage': train_mem_usage,
                    'train_loss': total_loss,
                    'loss': {v['step']:v['loss'] for v in loss_log if 'loss' in v},
                    # 'disk_w': disk_w,
                    # 'disk_t': disk_t,
                    'inf_data_len': inf_data_len,
                    'len_test': len_test,
                    'inf_run_time': inf_run_time,
                    'inf_run_time/n': inf_run_time/(len_test+0.001),
                    'inf_mem_usage': inf_mem_usage,
                    'calc_time': calc_time,
                    'calc_mem_usage': calc_mem_usage,
                }
                stats.append(stat)

        except:
            print(traceback.format_exc())
            try:
                if not keep_model:
                    del model
                gc.collect()
                del dataset
                gc.collect()
                del decoder
                gc.collect()
                if not keep_model:
                    del formatter
                gc.collect()
            except:
                pass

    if pp:
        with open(f"stats{gpu}.json","w") as f:
            json.dump(stats,f,indent=2)

        import pandas as pd
        
        # Convert to DataFrame
        df = pd.DataFrame(stats)
        
        # List of time-related fields
        time_fields = [
            'total_time',
            'prepare_run_time',
            'train_run_time',
            'inf_run_time',
            'inf_run_time/n',
            'calc_time'
        ]
        
        # Print total sum for each field
        print("\n=== Total Time Summary ===")
        print('of len',len(stats))
        for field in time_fields:
            if field in df.columns:
                total = df[field].sum()
                print(f"{field} Total: {round(total)} s. Ave:",round(total/len(stats)))
                
        mem_fields = ['train_mem_usage', 'inf_mem_usage', 'calc_mem_usage']
        
        print("\n=== Memory Usage Summary (GB) ===")
        for field in mem_fields:
            if field in df.columns:
                avg = df[field].mean()
                max_val = df[field].max()
                print(f"{field}: avg = {round(avg, 2)} GB, max = {round(max_val, 2)} GB")


class RemapCudaOOM:
    def __enter__(self): pass
    def __exit__(self, exc_type, exc_value, traceback):
        oom_errors = ["CUDA out of memory", "Make sure you have enough GPU RAM", "does not fit any GPU's remaining memory"]
        if exc_value and any(x in str(exc_value) for x in oom_errors):
            with open('submission.json', 'w') as f: f.write('cause submission scoring error')
